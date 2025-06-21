import torch
import torch.nn as nn
import torch.nn.functional as F

class VQ(nn.Module):
    def __init__(self, num_embeddings: int = 64, embed_dim: int = 16, commitment_cost: float = 0.25):
        """
        VQ class
        params num_embeddings: number of embeddings (K)
        params embed_dim: embedding dimension (D)
        params commitment_cost: commitment cost
        """
        super(VQ, self).__init__()
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

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

        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        commitment_loss = self.commitment_cost * commitment_loss

        return z_q, commitment_loss, codebook_loss

class ResBlock(nn.Module):
    def __init__(self, in_channels = 16, out_channels = 16, mode='encoder'):
        super(ResBlock, self).__init__()
        if(mode == 'encoder'):
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            )
    def forward(self, x):
        return self.block(x) + x
        
class Encoder(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = 16, n_layers=2):
        """
        Encoder class
        params hidden_channels: hidden channels
        params in_channels: input channels
        params n_layers: number of layers
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            *[ResBlock(in_channels=hidden_channels, out_channels=hidden_channels, mode='encoder') for _ in range(n_layers)],
        )
    
    def forward(self, x):
        """
        Forward pass
        params x: input data
        returns z_e: encoded data
        """
        return self.encoder(x)
        

class Decoder(nn.Module):
    def __init__(self, out_channels = 3, hidden_channels = 16, n_layers=2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            *[ResBlock(in_channels=hidden_channels, out_channels=hidden_channels, mode='decoder') for _ in range(n_layers)],
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        """
        Forward pass
        params x: input data
        returns x: decoded data
        """
        return self.decoder(x)


class VQVAE(nn.Module):
    def __init__(self, hidden_channels = 16, in_channels = 3, embedding_size = 16, n_layers=2):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, n_layers=n_layers)
        self.VQ = VQ(num_embeddings=embedding_size, embed_dim=hidden_channels)
        self.decoder = Decoder(out_channels=in_channels, hidden_channels=hidden_channels, n_layers=n_layers)

    def forward(self, x):
        """
        Forward pass
        params x: input data
        returns x: decoded data
        returns z_e: encoded data
        returns z_q: quantized data
        returns commitment_loss: commitment loss
        returns codebook_loss: codebook loss
        """
        z_e = self.encoder(x)
        z_q, commitment_loss, codebook_loss = self.VQ(z_e)
        x = self.decoder(z_q)
        return x, z_e, z_q, commitment_loss, codebook_loss