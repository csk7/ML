import torch
import torch.nn as nn
from utils.global_variables import GLOBAL_INFO
###Creating a VAE class###
class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, latent_size = 200, mode = 'visualize'):
        """
        VAE class
        params input_size: input size
        params hidden_size: hidden size
        params latent_size: latent size
        params mode: mode
        """
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, latent_size),
            nn.LeakyReLU(0.2),
        )

        if mode == 'visualize':
            self.meanLayer = nn.Linear(latent_size, 2)
            self.logVarLayer = nn.Linear(latent_size, 2)
        else:
            self.meanLayer = nn.Linear(latent_size, latent_size)
            self.logVarLayer = nn.Linear(latent_size, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(2 if mode == 'visualize' else latent_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x):    
        """
        Encode the input data
        params x: input data
        returns mean, logVar
        """
        x = self.encoder(x)
        mean = self.meanLayer(x)
        logVar = self.logVarLayer(x)
        return mean, logVar
        
    def decode(self, x):
        """
        Decode the latent space
        params x: latent space
        returns x_res: reconstructed data
        """
        x = self.decoder(x)
        return x
    
    def reparameterize(self, mean, logVar):
        """
        Reparameterize the latent space
        params mean: mean of the latent space
        params logVar: log variance of the latent space
        returns z: latent space
        """
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std).to(GLOBAL_INFO.device)
        return mean + eps * std

    def forward(self, x):
        """
        Forward pass
        params x: input data
        returns x_res: reconstructed data
        """
        mean, logVar = self.encode(x)
        z = self.reparameterize(mean, logVar)
        x_res = self.decode(z)
        return x_res, mean, logVar