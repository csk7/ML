from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels:int = 128, out_channels:int = 128):
        super(VAE_ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0) if(in_channels != out_channels) else nn.Identity(),
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        params x: input data (B, C_in, H, W)
        returns x: output data (B, C_out, H, W)
        """
        return self.block(x) + self.residual(x)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int = 32):
        super(VAE_AttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attn = SelfAttention(num_heads=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        params x: input data (B, C_in, H, W)
        returns x: output data (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        x = self.group_norm(x)
        x = x.view(B, C, H * W) #B,C, H, W -> B, C, H*W
        x = x.transpose(1, 2) #B, C, H*W -> B, H*W, C
        x = self.attn(x)
        x = x.transpose(1, 2) #B, H*W, C -> B, C, H*W
        x = x.view(B, C, H, W) #B, C, H*W -> B, C, H, W

        return x + residual

class Decoder(nn.Module):
    def __init__(self, scale_up_comp = 4):
        super(Decoder, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_AttentionBlock(channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp), #B, 512, H/8, W/8
            nn.Upsample(scale_factor=2), #B,512,H/4,W/4 Just Replicate

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=512//scale_up_comp), #B, 512, H/4, W/4
            nn.Upsample(scale_factor=2), #B,512,H/2,W/2 Just Replicate

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels=512//scale_up_comp, out_channels=256//scale_up_comp),
            VAE_ResidualBlock(in_channels=256//scale_up_comp, out_channels=256//scale_up_comp),
            VAE_ResidualBlock(in_channels=256//scale_up_comp, out_channels=256//scale_up_comp),
            VAE_ResidualBlock(in_channels=256//scale_up_comp, out_channels=256//scale_up_comp), #B, 256, H/2, W/2
            nn.Upsample(scale_factor=2), #B,256,H,W Just Replicate

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels=256//scale_up_comp, out_channels=128//scale_up_comp),
            VAE_ResidualBlock(in_channels=128//scale_up_comp, out_channels=128//scale_up_comp),
            VAE_ResidualBlock(in_channels=128//scale_up_comp, out_channels=128//scale_up_comp),
            VAE_ResidualBlock(in_channels=128//scale_up_comp, out_channels=128//scale_up_comp), #B, 128, H, W

            nn.GroupNorm(num_groups=32//scale_up_comp, num_channels=128//scale_up_comp),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, padding=1) #B, 3, H/4, W/4

        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        :params x:input (B, 4, H/8, W/8)
        :returns output: decoder output
        '''
        x /=  0.18215
        return self.block(x)