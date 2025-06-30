from turtle import forward
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, num_heads:int = 8, embed_dim: int = 32):
        super(SelfAttention, self).__init__()
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias = False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads

    def forward(self, x:torch.Tensor, causal_mask: bool=False) -> torch.Tensor:
        '''
        :params x: Input
        :params causal_mask: auto regressive property
        :returns output: Tensor output of forward pass
        '''
        B, T, C = x.shape

        #(B,T,C) @ (C, 3C) ---> B, T, 3C ; Each is B,T,C
        k, q, v  = self.in_proj(x).chunk(chunks = 3, dim = -1)

        #B,T,C ----> B,T,(C/H),H
        k = k.view(B, T, self.num_heads, self.head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        #B,T,(C/H),H  ---> B,(C/H),T,H
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        weights = torch.matmul(q, k.transpose(-1,-2)) / (self.head_dim ** 0.5)  #B,(C/H),T,H @ B,(C/H),H,T ---> B,(C/H),T,T
        if(causal_mask):
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1) 
            weights = weights.masked_fill(mask, -torch.inf)

        weights = weights.softmax(dim = -1)
        output = torch.matmul(weights, v) # B,(C/H),T,T @ B,(C/H),T,H ---> B,(C/H),T,H


        #B,(C/H),T,H --> B,T,(C/H),H ----> B,T,C
        output = output.transpose(1,2).reshape(B,T,C)

        return output


class CrossAttention(nn.Module):
    def __init__(self, num_heads:int, n_embed:int, n_embed_clip = 786):
        super(CrossAttention,self).__init__()
        self.query = nn.Linear(n_embed, n_embed, bias=True)
        self.key = nn.Linear(n_embed_clip, n_embed, bias = True)
        self.value = nn.Linear(n_embed_clip, n_embed, bias = True)
        self.num_heads = num_heads
        self.embed_per_head = n_embed // num_heads
        
    def forward(self, features, context):
        '''
        :params features: B, H*W, n_embed
        :params context: B, T, n_embed_clip T=77
        '''
        B, H_W, C = features.shape
        B, T, Cclip = context.shape
        q = self.query(features) #B, H*W, n_embed
        k = self.key(context) #B, T, n_embed
        v = self.value(context) #B, T, n_embed

        q=q.view(B,H_W,self.num_heads,self.embed_per_head).transpose(1,2) #B,nH,H_W,Ch
        k=k.view(B,T,self.num_heads,self.embed_per_head).transpose(1,2) #B,nH,T,Ch
        v=v.view(B,T,self.num_heads,self.embed_per_head).transpose(1,2) #B,nH,T,Ch

        weight = torch.matmul(q,k.transpose(-1,-2)) / (self.embed_per_head ** 0.5) #B,nH,H_W,T
        weight = weight.softmax(dim=-1)

        weight = torch.matmul(weight,v) #B,nH,H_W,Ch

        weight = weight.transpose(1,2).reshape(B,H_W,C)

        return weight






