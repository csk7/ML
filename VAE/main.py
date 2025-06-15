import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from model.model import VAE
from src.train import train
from inference.inference_functions import generate_digit, plot_latent_space
from utils.nn_model_tools import load_model
from utils.global_variables import GLOBAL_INFO
from utils.input_visual import show_input_data
from data.data import trainloader

###Model, loss function and optimizer###
#model
model = VAE(mode='visualize').to(GLOBAL_INFO.device)
#optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#Loss function
def loss_function(x, x_res, mean, logVar):
    """
    Loss function for VAE
    params x: input data
    params x_res: reconstructed data
    params mean: mean of the latent space
    params logVar: log variance of the latent space
    returns loss: loss
    """
    BCE = F.binary_cross_entropy(x_res, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())
    return BCE + KLD

###Training###
show_input_data(file_location=os.path.join(GLOBAL_INFO.script_dir, 'img', 'mnist.png'))
train(model, trainloader, optimizer, loss_function, epochs=1, save_path=os.path.join(GLOBAL_INFO.script_dir, 'vae_model.pth'))

# Inference
loaded_model, _, _, _ = load_model(model, optimizer, path=os.path.join(GLOBAL_INFO.script_dir, 'vae_model.pth'))
generate_digit(loaded_model, 0.0, 1.0, num_samples=1, latent_size=2, file_name=os.path.join(GLOBAL_INFO.script_dir, 'img', 'generated_after_loading.png'))
plot_latent_space(loaded_model, scale=1.0, n=25, digit_size=28, figsize=15, file_name=os.path.join(GLOBAL_INFO.script_dir, 'img', 'latent_space.png'))

    