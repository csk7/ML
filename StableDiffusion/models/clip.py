import torch
import torch.nn as nn

from models.attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size:int = 49408, embedding_size:int = 768, context_length:int = 77):
        super(CLIPEmbedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_embed = nn.Parameter(torch.zeros(context_length, embedding_size))
        self.context_length = context_length

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        '''
        :params x: input word index as torch.LongTensor of size B, T
        :returns output: torch.FloatTensor of size B, T, C
        '''
        x = self.token_embed(x) + self.pos_embed # B,T,C + T,C
        return x

class CLIPLayer(nn.Module):
    def __init__(self, num_heads = 12, embedding_size = 768):
        super(CLIPLayer, self).__init__()
        
        self.ln_1 = nn.LayerNorm(embedding_size)
        self.mha = SelfAttention(num_heads=num_heads, embed_dim= embedding_size) #B, T, C ---> B, T, C

        self.ln_2 = nn.LayerNorm(embedding_size)
        self.ff1 = nn.Linear(embedding_size, embedding_size*4) 
        self.ff2 = nn.Linear(embedding_size*4, embedding_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :params x: (B,T,C)
        :returns output: (B,T,C)
        '''
        x = x + self.mha(self.ln_1(x), True) #B, T, C ----> B, T, C
        residual = x 
        x = self.ff1(self.ln_2(x)) #B, T, C ----> B, T, 4*C
        x = self.ff2(torch.sigmoid(1.702*x)) #QuickGeLU ----> B, T, C
        x = x + residual
        return x


class CLIP(nn.Module):
    def __init__(self, num_layers:int = 12, num_heads:int = 12):
        super(CLIP, self).__init__()
        self.embedding = CLIPEmbedding(vocab_size = 49408, embedding_size = 768, context_length = 77) #Pretraining
        self.block = nn.Sequential(
            *[CLIPLayer(num_heads = 12, embedding_size = 768) for _ in range(num_layers)],
        )

        self.lastNorm = nn.LayerNorm(768)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        '''
        :params x: input tensor (B,T)
        :returns output: (B,T,C)
        '''

        state = self.embedding(x) #B,T ---> B,T,C
        output = self.block(state) #B,T,C --> B,T,C
        output = self.lastNorm(output) #B,T,C --> B,T,C

        return output
