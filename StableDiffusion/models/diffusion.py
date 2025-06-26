from importlib import invalidate_caches
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention, CrossAttention
#----------------Time Embedding---------------
class TimeEmbeddding(nn.Module):
    def __init__(self, n_dim :int = 32):
        super(TimeEmbeddding,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_dim, 4*n_dim),
            nn.SiLU(),
            nn.Linear(4*n_dim , dim)
        )

    def forward(self, x):
        x = self.block(x)
        return x



#---------------Unets-------------------
class SwitchSequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self. block = nn.Sequential(
            *args
        )

    def forward(self, x:torch.Tensor, context: torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        '''
        :params x: (B,4,H/8,W/8)
        :params context: (B, T, C)
        :params time: (1, 1280)
        :returns output: (B,320,H/8,W/8)
        '''
        for layer in self.block:
            if(isinstance(layer,UnetAttentionBlock)):
                x = layer(x, context)
            elif(isintance(layer,UnetResidualBlock)):
                x = layer(x, time)
            else:
                x = layer(x)

        return x

class Upsample(nn.Module):
    def __init__(self, channels : int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1)


    def forward(self, x):
        '''
        :params x: B, C, H, W ---> B, C, 2H, 2W
        '''
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class UnetOutput(nn.Module):
    def __init__(self, in_channels:int = 320, out_channels:int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 3, padding=1)
        )

    def forward(self, x):
        '''
        :params x: B, 320, H/8, W/8 ---> B,4,H/8,W/8
        '''
        return self.block(x)


class UnetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int, n_time:int = 1280):
        super().__init()
        self.FeatureBlock = nn.Sequential(
            nn.GroupNorm(num_groups = 32, num_channels = in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding = 1)
        )
        self.TimeBlock = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_dim, out_channels)
        )
        self.MergedBlock = nn.Sequential(
            nn.GroupNorm(num_groups = 32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )

        if(n_channels!=out_channels):
            self.residualLayer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        else:
            self.residualLayer = nn.Identity()

    def forward(self, feature:torch.Tensor, time:torch.Tensor):
        '''
        :params feature: B. Cin, H, W
        :params time   : 1,n_dim
        :returns output: B, Cout, H, W
        '''
        feature_out = self.FeatureBlock(feature) #B,Cin,H,W ---> B,Cout,H,W
        time_out = self.TimeBlock(time) #1,n_dim ---> 1,Cout
        merged_out = feature_out + time_out.unsqueeze(-1).unsqueeze(-1) # B,Cout,H,W + 1,Cout,1,1 -->  B,Cout,H,W
        return self.residualLayer(feature) + self.MergedBlock(merged_out) # B,Cout,H,W -->  B,Cout,H,W


class UnetAttentionBlock(nn.Module):
    def __init__(self, num_heads:int, n_embed:int, d_embed = 768): 
        super().__init__()
        channels = num_heads*n_embed
        
        self.init_block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        )

        self.self_attention_block = nn.Sequential(
            nn.LayerNorm(channels),
            SelfAttention(num_heads=num_heads, embed_dim=n_embed)
        )

        self.cross_attention_block = nn.Sequential(
            nn.LayerNorm(channels),
            CrossAttention(num_heads=num_heads, n_embed = n_embed, d_embed_clip = d_embed)
        )

        self.FFN_1 = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 4*channels*2),
        )
        self.FFN_2 = nn.Linear(4*channels, channels)
        

        self.end_block = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, features:torch.Tensor, context:torch.Tensor) -> torch.Tensor:
        '''
        :params features: B,C,H,W
        :params context: B,T,C_embed
        '''
        residual_long = features
        features = self.init_block(features)

        B, C, H, W = features.shape
        residual_short = features.view(B,C,H*W).transpose(-1,-2) #B,C,H,W --> B,H*W,C
        features = self.self_attention_block(features)
        features = residual_short + features
        
        residual_short = features
        features = self.cross_attention_block(features, context)
        features = residual_short + features
        
        residual_short = features
        features = self.FFN_1(features)
        features, gate = features.chunk(2, dim=-1)
        features = features * F.gelu(gate)
        features = self.FFN_2(features)
        features = residual_short + features

        features = features.transpose(-1,-2).reshape(B,C,H,W)
        
        features = self.end_block(features) + residual_long
        return features


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.encoder = nn.ModuleList([
            #B, 4, H/8, W/8 --> B, 320, H/8, W/8
            SwitchSequential(nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)),
            SwitchSequential(UnetResidualBlock(in_channels = 320, out_channels = 320), UnetAttentionBlock(num_heads=8, dim = 40)),
            SwitchSequential(UnetResidualBlock(in_channels = 320, out_channels = 320), UnetAttentionBlock(num_heads=8, dim = 40)),

            #B, 320, H/8, W/8 ---> B, 640, H/16, W/16
            SwitchSequential(nn.Conv2d(in_channels=320, out_channels=320, stride = 2, kernel_size=3, padding=1)),
            SwitchSequential(UnetResidualBlock(in_channels = 320, out_channels = 640), UnetAttentionBlock(num_heads=8, dim = 80)),
            SwitchSequential(UnetResidualBlock(in_channels = 640, out_channels = 640), UnetAttentionBlock(num_heads=8, dim = 80)),

            #B, 640, H/16, W/16 ---> B, 1280, H/32, W/32
            SwitchSequential(nn.Conv2d(in_channels=640, out_channels=640, stride = 2, kernel_size=3, padding=1)),
            SwitchSequential(UnetResidualBlock(in_channels = 640, out_channels = 1280), UnetAttentionBlock(num_heads=8, dim = 160)),
            SwitchSequential(UnetResidualBlock(in_channels = 1280, out_channels = 1280), UnetAttentionBlock(num_heads=8, dim = 160)),

            #B, 1280, H/32, W/32 ---> B, 1280, H/64, W/64
            SwitchSequential(nn.Conv2d(in_channels=1280, out_channels=1280, stride = 2, kernel_size=3, padding=1)),
            SwitchSequential(UnetResidualBlock(in_channels = 1280, out_channels = 1280)),
            SwitchSequential(UnetResidualBlock(in_channels = 1280, out_channels = 1280)),
        ])

        self.bottleneck = SwitchSequential(
            #B, 1280, H/64, W/64 ---> B,1280,H/64,W/64
            UnetResidualBlock(1280, 1280),
            UnetAttentionBlock(8, 160),
            UnetResidualBlock(1280, 1280)
        )

        self.Decoder = nn.ModuleList(
            #B,1280,H/64,W/64 ; B,1280,H/64,W/64 ---> B, 1280, H/32, W/32
            SwitchSequential(UnetResidualBlock(in_channels = 2560, out_channels = 1280)),
            SwitchSequential(UnetResidualBlock(in_channels = 2560, out_channels = 1280)),
            SwitchSequential(UnetResidualBlock(in_channels = 2560, out_channels = 1280), Upsample(n_channels = 1280)),

            #B, 1280, H/32, W/32; B, 1280, H/32, W/32 ---> B, 1280, H/32, W/32
            SwitchSequential(UnetResidualBlock(in_channels = 2560, out_channels = 1280), UnetAttentionBlock(num_heads=8, dim = 160)),
            SwitchSequential(UnetResidualBlock(in_channels = 2560, out_channels = 1280), UnetAttentionBlock(num_heads=8, dim = 160)),
            SwitchSequential(UnetResidualBlock(in_channels = 1920, out_channels = 1280), UnetAttentionBlock(num_heads=8, dim = 160), Upsample(n_channels = 1280)),

            #B, 1280, H/32, W/32; B, 1280, H/32, W/32 ---> B, 640, H/16, W/16
            SwitchSequential(UnetResidualBlock(in_channels = 1920, out_channels = 640), UnetAttentionBlock(num_heads=8, dim = 80)),
            SwitchSequential(UnetResidualBlock(in_channels = 1280, out_channels = 640), UnetAttentionBlock(num_heads=8, dim = 80)),
            SwitchSequential(UnetResidualBlock(in_channels = 960, out_channels = 640), UnetAttentionBlock(num_heads=8, dim = 80), Upsample(n_channels = 640)),

            #B, 640, H/16, W/16; B, 640, H/16, W/16 ---> B, 320, H/8, W/8
            SwitchSequential(UnetResidualBlock(in_channels = 960, out_channels = 320), UnetAttentionBlock(num_heads=8, dim = 40)),
            SwitchSequential(UnetResidualBlock(in_channels = 640, out_channels = 320), UnetAttentionBlock(num_heads=8, dim = 40)),
            SwitchSequential(UnetResidualBlock(in_channels = 640, out_channels = 320), UnetAttentionBlock(num_heads=8, dim = 40), Upsample(n_channels = 320)),

        )
#--------------------------------------

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbeddding(in_size = 320) #1,320 ---> 1,1280
        self.Unet = UNET() #Latent_Noise, Time
        self.Unet_output = UnetOutput(320, 4) #To go from output of Unet size to noise size

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        '''
        :params x: Lantent representation of noise B, 4, H/8, W/8
        :params context: CLIP output B, T, C (T=77, C=768)
        :params time: Number through embedding layer output provided - time embedding 1, 320. 
        :output: Noise prediction of each step B,  4, H/8, W/8
        '''

        x = self.time_embedding(x) #(1,320) ---> (1,1280)
        x = self.Unet(x, context, time) #(B, 4, H/8, W/8); (B, T, C), (1,1280) ---> (B, 320, H/8, W/8)
        x = self.Unet_output(x)

        return x


