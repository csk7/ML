import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, scale_down_comp = 4):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            #B, in_channels, H, W --> B, out_channels, H, W
            nn.Conv2d(in_channels=3, out_channels=128//scale_down_comp, kernel_size=3, stride=1, padding=1),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp),        

            #B, out_channels, H, W --> B, 2*out_channels, H/2, W/2    
            nn.Conv2d(in_channels=128//scale_down_comp, out_channels=128//scale_down_comp, kernel_size=3, stride=2, padding=0),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=128//scale_down_comp, out_channels=256//scale_down_comp),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp),

            #B, out_channels, H, W --> B, 4*out_channels, H/4, W/4    
            nn.Conv2d(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp, kernel_size=3, stride=2, padding=0),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=256//scale_down_comp, out_channels=256//scale_down_comp),

            #B, out_channels, H, W --> B, 8*out_channels, H/2, W/2    
            nn.Conv2d(in_channels=512//scale_down_comp, out_channels=1024//scale_down_comp, kernel_size=3, stride=2, padding=0),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=1024//scale_down_comp, out_channels=1024//scale_down_comp),
            #B, out_channels, H, W --> B, out_channels, H, W
            VAE_ResidualBlock(in_channels=1024//scale_down_comp, out_channels=1024//scale_down_comp),
        )