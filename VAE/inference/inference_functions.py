import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.global_variables import GLOBAL_INFO

def generate_digit(model, mean, variance, num_samples = 1, latent_size = 200, file_name='mnist_generated.png'):
    """
    Generate a digit from the latent space
    params model: model
    params mean: mean of the latent space
    params variance: variance of the latent space
    params num_samples: number of samples
    params latent_size: latent size
    params file_name: file name
    """
    std = torch.sqrt(torch.tensor(variance, dtype = torch.float).to(GLOBAL_INFO.device))
    normalDistribution = torch.distributions.Normal(mean, std)
    z_sample = normalDistribution.sample((num_samples, latent_size))
    z_sample = z_sample.to(GLOBAL_INFO.device)
    
    model.eval()
    x_decoded = model.decode(z_sample)
    model.train()
    
    digit = x_decoded.detach().cpu().reshape(28,28)
    plt.imshow(digit, cmap ='gray')
    plt.axis('off')
    
    plt.savefig(file_name)
    plt.close()

def plot_latent_space(model, scale = 1.0, n=25, digit_size=28, figsize=15, file_name='latent_space.png'):
    """
    Plot the continuous 2d latent space
    params model: model
    params scale: scale
    params n: number of samples
    params digit_size: digit size
    params figsize: figure size
    params file_name: file name
    """
    figure = np.zeros((digit_size*n, digit_size*n))
    val_y = np.linspace(-scale, scale, n)
    val_x = np.linspace(-scale, scale, n)
    for i, y_i in enumerate(val_y):
        for j, x_j in enumerate(val_x):
            z = torch.tensor([x_j, y_i], dtype=torch.float).unsqueeze(0).to(GLOBAL_INFO.device)
            x_decoded = model.decode(z)
            digit = x_decoded.detach().cpu().reshape(28,28)
            figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = digit
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.savefig(file_name)
    plt.close()