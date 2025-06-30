import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder import VAE_ResidualBlock, VAE_AttentionBlock
class Encoder(nn.Module):
    def __init__(self, scale_down_comp = 1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            
            #B, C_in, H, W --> B, C, H, W
            nn.Conv2d(in_channels=3, out_channels=128//scale_down_comp, kernel_size=3, stride=1, padding=1), 
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp), #B, C, H, W --> B, C, H, W
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp), #B, C, H, W --> B, C, H, W

            #B, C, H, W --> B, C, H/2, W/2    
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=256//scale_down_comp), #B, C, H/2, W/2 --> B, 2*C, H/2, W/2
            VAE_ResidualBlock(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp), #B, 2*C, H/2, W/2 --> B, 2*C, H/2, W/2

            #B, 2*C, H/2, W/2 --> B, 2*C, H/4, W/4    
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(in_channels=256//scale_down_comp, out_channels=512//scale_down_comp), #B, 2*C, H/4, W/4 --> B, 4*C, H/4, W/4        
            VAE_ResidualBlock(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp), #B, 4*C, H/4, W/4 --> B, 4*C, H/4, W/4

            #B, 4*C, H/4, W/4 --> B, 4*C, H/8, W/8    
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            VAE_ResidualBlock(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            VAE_ResidualBlock(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            
            VAE_AttentionBlock(channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            VAE_ResidualBlock(in_channels=512//scale_down_comp, out_channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            
            nn.GroupNorm(num_groups=32//scale_down_comp, num_channels=512//scale_down_comp), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            nn.SiLU(), #B, 4*C, H/8, W/8 --> B, 4*C, H/8, W/8
            
            nn.Conv2d(in_channels=512//scale_down_comp, out_channels=8, kernel_size=3, padding=1), #B, 4*C, H/8, W/8 --> B, 8, H/8, W/8
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), #B, 8, H/8, W/8 --> B, 8, H/8, W/8
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        params x: input data (B, C_in, H, W)
        params noise: noise (B, C_out, H/8, W/8)
        returns x: output data (B, C_out, H/8, W/8)
        """
        x = self.encoder(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1) #B, 8, H/8, W/8 --> B, 4, H/8, W/8; B, 4, H/8, W/8
        log_variance = torch.clamp(log_variance, min=-20, max=20)
        variance = torch.exp(log_variance)
        std = torch.sqrt(variance)

        #N(0,1) -> N(mean, std)    
        z = mean + std * noise

        #Scaling Constant
        z = z * 0.18215
        return z
