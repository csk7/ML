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

        weights = torch.matmul(q, k.transpose(-1,-2)) // torch.sqrt(self.head_dim)  #B,(C/H),T,H @ B,(C/H),H,T ---> B,(C/H),T,T
        if(causal_mask):
            self.register_buffer("mask", torch.triu(torch.ones_like(input = weights, device=device),diagonal=1))
            weights = weights.masked_fill(mask, -torch.inf)

        weights = weights.softmax(dim = -1)
        output = torch.matmul(weights, v) # B,(C/H),T,T @ B,(C/H),T,H ---> B,(C/H),T,H


        #B,(C/H),T,H --> B,T,(C/H),H ----> B,T,C
        output = output.transpose(1,2).reshape(B,T,C)

        return output



