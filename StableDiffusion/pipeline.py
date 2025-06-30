from ast import Tuple
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import tqdm
from ddpm import ddpm_sampler

#Global Parameters#
HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = 512//8
LATENT_WIDTH = 512//8

def generate(prompt:str, uncond_prompt:str, input_image=None, strength=0.1, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm",
    n_inference_steps=50, models={},seed=None, device=None, idle_device=None, tokenizer=None):

    with torch.no_grad():
        #Make Sure strength is between 0 and 1
        if(not(0< strength <=1)):
            raise ValueError("Strength miust be between 0 and 1")

        #Create a function to sheift models to idle device
        if(idle_device):
            to_idle = lambda x:x.to(idle_device)
        else:
            to_idle = lambda x:x

        #Reproducability
        generator = torch.Generator()
        if(seed is not None):
            generator.manual_seed(seed)
        else:
            generator.seed()

        #For Text to Image and Image to Image - Process input prompts and  get output of CLIP
        #CLIP
        clip = models['clip']
        clip.to(device)
        #CFG -> Weights Prompt and negative prompt (empty) requires 2 passes. Without cfg no control over how much flow? 
        if(do_cfg):
            #Convert prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids #1,T
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device=device)
            cond_context = clip(cond_tokens) #1,T,C
            #Convert Negative prompt to tokens
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids #I,T
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context],dim=0) #(1,T,C) , (1,T,C) --> (2, T, C)

        else:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.LongTensor, device=device) #1,T
            context = clip(cond_tokens) #1,T,C

        to_idle(clip)

        #Now we need to sample from the random noise distribution
        if(sampler_name == "ddpm"):
            sampler = ddpm_sampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sample not found{sampler_name}")

        #No image then generate from random noise, If image given then we need to create noisy image from given image
        latent_shape = (1,4,LATENT_HEIGHT,LATENT_WIDTH)
        if(input_image is not None):
            encoder = models['encoder']
            encoder = encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT)) #image obj
            input_image_tensor = np.array(input_image_tensor) #numpy
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32) #numpy to tensor
            input_image_tensor = rescale(input_image_tensor, (0,255),(-1,1)) #rescale tensor H,W,C
            input_image_tensor = torch.unsqueeze(input_image_tensor,0) #1,H,W,C
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            #Randomly sampling from the latent space(reparametrization trick)
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            latent_image = encoder(input_image_tensor, encoder_noise) #We sampled from the latent distribution based on the encoder noise

            #The sampler must know how much noise should be added every step in DDPM. If strenght = 1 then Fully noise added. More creative model.
            sampler.set_strength(strength) 
            latent_image = sampler.add_noise(latent_image, sampler.timesteps[0])
            to_idle(encoder)

        else:
            #Doing text to image; latents is pure noise
            latent_image = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion = diffusion.to(device)

        #We need to prepare the Time steps
        #There is total time steps, n_inference steps, If total_training_steps = 1000 and inference step = 50, Then we do DDPM for 1000, 980, 960..0
        #If both are 50 then we do for 50,49,48...1. so timesteps in above cases = [1000, 980...1]. Timesteps encodes and put into embedding. Essentially we jump and try to predict all the noise

        timesteps = tqdm.tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #Time Embedding - A Number to vector of [1,320]
            time_embedding = get_time_embedding(timestep).to(device)

            #Now prepare latents (B, 4, Latent_height, Latent_Weight)
            model_inputs = latent_image

            if(do_cfg):
                #We need to add same latents twice for with cond_context and with un_cond_context
                model_inputs = model_inputs.repeat(2, 1, 1, 1) #(B,T,H/4,W/4) --> (2*B,T,H/4,W/4)
                
            #Model output is predict noise 
            model_output = diffusion(x = model_inputs, context = context, time = time_embedding) #(2*B,T,H/4,W/4)

            if(do_cfg):
                #We need to separate the 2 outputs if we provided both
                cond_output, uncond_output = model_output.chunk(2, dim = 0) #(2*B,T,H/4,W/4) --> (B,T,H/4,W/4), (B,T,H/4,W/4)
                model_output = cfg_scale*(cond_output - uncond_output) + uncond_output #This is the predicted noise in the time step

            #We need to remove this noise predicted
            latent_image = sampler.step(timestep, latent_image, model_output)

        to_idle(diffusion)

        #Decoder to get the iamge from the latents
        decoder = models['decoder']
        decoder = decoder.to(device)

        images = decoder(latent_image)
        to_idle(decoder)

        images = rescale(images, (-1,1), (0,255))
        images = images.squeeze(0).permute(1,2,0)
        images = images.to('cpu', torch.uint8).numpy() #need to convert it to H,W,C for CPU
        return images

def rescale(x: torch.Tensor, old_range:Tuple, new_range:Tuple, clamp=False) -> torch.Tensor:
    '''
    scales the tensor from old range to new range
    '''
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = x-old_min
    scale = (new_max - new_min)/(old_max - old_min)
    x = x*scale
    x = x+new_min        
    if(clamp):
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep: int) ->torch.Tensor:
    freq = torch.pow(10000,-torch.arange(start = 0, end =160, dtype=torch.float32)/160) #(160)
    output = torch.tensor([timestep], dtype = torch.float32) * freq #(160)
    output = torch.unsqueeze(output,0) #1,160
    return torch.cat([torch.cos(output), torch.sin(output)], dim=-1) #(1,320)
















        
                    
