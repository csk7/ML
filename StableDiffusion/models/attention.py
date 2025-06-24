from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, embed_dim:int = 32):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.key = nn.linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        forward pass of self attention
        :params x: input
        :returns output:self attention output
        '''
        B, T, C = x #T=H*W All pixels in context; embed_dim = Channels (c)
        k = self.key(x)
        q = self.value(x)
        v = self.query(x)

        wei = torch.matmul(q, k.transpose())//torch.sqrt(self.embed_dim) # B, T, C * B, C, T --> B, T, T
        wei = F.softmax(wei, dim =-1)
        wei = torch.matmul(wei, v) #B, T, T * B, T, C --> B, T, C

        return wei

class MultiHeadSA(nn.Module):
    def __init__(self, num_heads:int = 8, embed_dim = 32):
        super(MultiHeadSA, self).__super__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim //self.num_heads
        self.block = nn.ModuleList([SelfAttention(embed_dim=self.head_dim) for _ in range(num_heads)])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        forward function

        :params x: Input Tensor
        :returns out:Output Tensor
        '''
        return torch.cat(*([sa(x) for sa in self.block]), dim = -1)



