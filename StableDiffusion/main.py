from models.clip import CLIP
from models.encoder import Encoder
from models.decoder import Decoder
from models.diffusion import Diffusion
import model_convertor

###Model Loader####
def preload_model_from_standard_weights(ckt_path, device):
    state_dict = model_convertor.load_from_standard_weights(ckt_path, device)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict=state_dict['encoder'], strict=False)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict=state_dict['decoder'], strict=False) #By Name of the labels loaded. Names do not match. 

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict=state_dict['diffusion'], strict=False)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict=state_dict['clip'], strict=False)

    return {
        'encoder' : encoder,
        'decoder' : decoder,
        'diffusion' : diffusion,
        'clip' : clip
    }

###Main###
import os
import torch
import pipeline
from PIL import Image
from transformers import CLIPTokenizer

if __name__ == '__main__':
    MAC_OS = False
    NVDIA_GPU = False
    DEVICE = 'cpu'
    if(NVDIA_GPU):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(MAC_OS):
        DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'DEVICE : {DEVICE}')

    current_path = os.path.dirname(__file__)
    tokenizer = CLIPTokenizer(os.path.join(current_path,'data','vocab.json'), merges_file = os.path.join(current_path,'data','merges.txt'))
    model_file = os.path.join(current_path,'data','v1-5-pruned-emaonly.ckpt')
    models = preload_model_from_standard_weights(model_file, DEVICE)

    #Text to Image#
    prompt = "Add reading glasses to Dog"
    uncond_prompt = ""
    do_cfg=True
    cfg_scale = 7

    #Image to Image#
    input_image = None
    image_path = os.path.join(current_path,'data','Cute_dog.jpg')
    input_image = Image.open(image_path)
    strength = 0.9

    sampler = "ddpm"
    num_reference_steps = 50
    seed = 48

    output_image = pipeline.generate(prompt = prompt, 
        uncond_prompt = uncond_prompt, 
        input_image = input_image, 
        strength=strength, 
        do_cfg=do_cfg, 
        cfg_scale=cfg_scale, 
        sampler_name=sampler,
        n_inference_steps=num_reference_steps, models=models,seed=seed, device=DEVICE, idle_device='cpu', tokenizer=tokenizer)

    output_image_save = Image.fromarray(output_image)
    output_image_save.save(os.path.join(current_path,'data','output_image.png'))
