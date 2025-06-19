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



class VQ(nn.Module):
    def __init__(self, k: int, d: int):
        super(VQ, self).__init__()
        self.embeddings = nn.Embedding(k, d)
        
    def forward(self, x):
        """
        Forward pass
        params x: input data
        """
        z_e = x
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C) 
        distance = torch.sum(x_flat**2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight**2, dim=1, keepdim=True).t() \
            - 2 * torch.matmul(x_flat, self.embeddings.weight.t())
        argmin = torch.argmin(distance, dim=1)
        z_q = self.embeddings(argmin)
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)

        z_e = z_q + (z_e - z_q).detach()
        return z_e, z_q
        
        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.nnLayers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.nnLayers(x) + x

class AllResBlocks(nn.Module):
    def __init__(self, n_layers):
        super(AllResBlocks, self).__init__()
        self.nnLayers = nn.Sequential(
            *[ResBlock(in_channels=16, out_channels=16) for _ in range(n_layers)]    
        )
    
    def forward(self, x):
        return self.nnLayers(x)

class VQVAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, latent_size = 200, mode = 'visualize'):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            AllResBlocks(n_layers=2),
        )
        self.VQ = VQ(10, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass
        params x: input data
        """
        z_e = self.encoder(x)
        z_q = self.VQ(z_e)
        x = self.decoder(z_q)
        return x, z_e, z_q
