import torch
import torch.nn as nn
import torch.nn.functional as F

#Global Parameters#
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads:int =8, n_embed = 128):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_dim = n_embed // n_heads
        self.fc_out = nn.Linear(n_embed, n_embed)
        self.register_buffer("mask", torch.tril(torch.ones((n_embed, n_embed), device = device)))

    def forward(self, v:torch.Tensor, k:torch.Tensor, q:torch.Tensor) -> torch.Tensor:
        '''
        :params values: V in attention (B, T, C)
        :params keys: K in attention (B, T, C)
        :params query: Q in attention (B, T, C)
        :return: attention output (B, T, C)
        '''
        B, T, C = v.shape

        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)# (B, T, C) -> (B, n_h, T, d_h)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)# (B, T, C) -> (B, n_h, T, d_h)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)# (B, T, C) -> (B, n_h, T, d_h)

        weights = torch.matmul(q, k.transpose(-2, -1)) // (self.head_dim ** 0.5)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = torch.matmul(weights, v) # (B, n_h, T, d_h)
        weights = weights.transpose(1,2).reshape(B, T, C) # (B, T, C)
        weights = self.fc_out(weights)
        return weights

        
class MulitHeadLatentAttention(nn.Module):
    def __init__(self, n_heads:int =8, n_embed = 128, d_lora_kv:int = 32, d_lora_q:int = 32, d_rope_kq:int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_dim = n_embed // n_heads
        self.d_lora_kv = d_lora_kv
        self.d_lora_q = d_lora_q
        self.d_rope_kq = d_rope_kq
        
        self.Wc_kv = nn.Linear(n_embed, d_lora_kv)
        self.Wk_c = nn.Linear(d_lora_kv, n_embed)

        self.Wq_v = nn.Linear(d_lora_q, n_embed)

        self.Wc_q = nn.Linear(n_embed, d_lora_q)
        self.Wq_c = nn.Linear(d_lora_q, n_embed)

        self.sin_table, self.cos_table = GetSinCosTable(d_rope_kq, d_rope_kq)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        :params x: (B, T, C)
        :return: (B, T, C)
        '''
        B, T, C = x.shape
        c_kv = self.Wc_kv(x) # (B, T, d_r)
        c_q = self.Wc_q(x) # (B, T, d_r)
        
        k = self.Wk_c(c_kv) # (B, T, C)
        q = self.Wq_c(c_q) # (B, T, C)
        
        k = ApplyRope(x.view(B, T, self.n_heads, self.head_dim), self.sin_table, self.cos_table)
        q = ApplyRope(c_q.view(B, T, self.n_heads, self.head_dim), self.sin_table, self.cos_table)
        
        v = self.Wq_v(c_q) # (B, T, C)


def GetSinCosTable(seq_len:int, d_model:int, device:str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    seq_dim = torch.arange(seq_len, device = device).unsqueeze(1) # (T, 1)
    embed_dim = torch.pow(10000, -2*(torch.arange(0, d_model//2, device = device))/d_model)    # (d_model/2)
    embed_dim = embed_dim.unsqueeze(0) # (1, d_model/2)
    sin_table = torch.sin(seq_dim @ embed_dim) # (T, d_model/2)
    cos_table = torch.cos(seq_dim @ embed_dim) # (T, d_model/2)
    return sin_table, cos_table
    

def ApplyRope(x: torch.Tensor) -> torch.Tensor:
    '''
    Apply rope based on the Position in dim =1
    :params x: (B, T, nH, dH    )
    :return: (B, T, nH, dH)
    '''
    B, T, nH, dH = x.shape
    x = x.reshape(B, T, nH, dH//2, 2)
    sin_table, cos_table = GetSinCosTable(T, dH)
    x1, x2 = x[:, :, :, :, 0], x[:, :, :, :, 1] # (B, T, nH, dH//2), (B, T, nH, dH//2)
    sin_table = sin_table.unsqueeze(0).unsqueeze(2) # (1, T, 1, dH//2)
    cos_table = cos_table.unsqueeze(0).unsqueeze(2) # (1, T, 1, dH//2)
    x1 = x1 * cos_table - x2 * sin_table # (B, T, nH, dH//2)
    x2 = x2 * cos_table + x1 * sin_table # (B, T, nH, dH//2)
    x = torch.stack((x1, x2), dim = -1) # (B, T, nH, dH//2, 2)
    x = x.reshape(B, T, nH, dH)
    return x
    
    

    
        
        
        

        