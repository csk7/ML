from models.clip import CLIP
from models.encoder import Encoder
from models.decoder import Decoder
from models.diffusion import diffusion
import model_convertor

###Model Loader####
def preload_model_from_standard_weights(ckt_path, device):
    state_dict = model_convertor.load_from_standard_weights(ckt_path, device)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict=state_dict['encoder'], strict=True)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict=state_dict['decoder'], strict=True) #By Name of the labels loaded. Names do not match. 

    diffusion = diffusion().to(device)
    diffusion.load_state_dict(state_dict=state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict=state_dict['clip'], strict=True)

    return {
        'encoder' : encoder,
        'decoder' : decoder,
        'diffusion' : diffusion,
        'clip' : clip
    }

###Main###
import torch
import pipeline
from PIL import Image
from transformers import CLIPTokenizers

if __name__ == '__main__':
    MAC_OS = False
    NVDIA_GPU = True
    if(NVDIA_GPU):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(MAC_OS):
        DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'DEVICE : {DEVICE}')

    tokenizer = CLIPTokenizers('../data/vocab.json', merges_file = "../data/merges.py")
    model_file = '../data/v1-5-pruned-emaonly.ckpt'
    models = preload_model_from_standard_weights(model_file, DEVICE)

    #Text to Image#
    prompt = "Add reading glasses to Dog"
    uncond_promt = ""
    do_cfg=True
    cfg_scale = 7

    #Image to Image#
    input_image = None
    image_path = "../data/Cute_dog.jpg"
    #input_image = Image.open(image_path')
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
    output_image_save.save("../data/output_image.png")
    