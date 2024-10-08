from torch import nn
import torch
import matplotlib.pyplot as plt
import math

from attention import MultiHeadSelfAttentionLayer

class FeedForward(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, dim_out: int , dtype=torch.float32):
        super(FeedForward, self).__init__()
        self.dtype=dtype
        self.ff_1 = nn.Linear(dim_in, dim_ff, bias=True, dtype=self.dtype)
        self.activation = nn.GELU()
        self.ff_2 = nn.Linear(dim_ff, dim_out, bias=True, dtype=self.dtype)
    def forward(self, x):
        return self.ff_2(self.activation(self.ff_1(x)))


class ClassificationHead(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, num_classes: int , dtype=torch.float32):
        super(ClassificationHead, self).__init__()
        self.dtype=dtype
        self.ff_1 = nn.Linear(dim_in, dim_ff, bias=True, dtype=self.dtype)
        self.activation = nn.Tanh() # tanh activation
        self.ff_2 = nn.Linear(dim_ff, num_classes, bias=True, dtype=self.dtype)

    def forward(self, x):
        return self.ff_2(self.activation(self.ff_1(x))) # B, N, num_classes
    

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, dim_ff: int, num_heads: int, context_length: int, dtype=torch.float32):
        super(TransformerBlock, self).__init__()
        self.dtype=dtype
        self.layer_norm_1 = nn.LayerNorm(emb_dim, dtype=self.dtype) # only normalize over embedding dim.
        self.layer_norm_2 = nn.LayerNorm(emb_dim, dtype=self.dtype)
        self.multi_head_att = MultiHeadSelfAttentionLayer(
            dim_in=emb_dim, dim_out=emb_dim, num_heads=num_heads, context_length=context_length, dtype=self.dtype
        )
        self.feed_forward = FeedForward(dim_in=emb_dim, dim_ff=dim_ff, dim_out=emb_dim, dtype=self.dtype)
    def forward(self, x):
        x_prime = x + self.multi_head_att(x)#self.layer_norm_1(x))
        return x_prime+self.feed_forward(x) #self.layer_norm_2(x_prime))
    
class PositionEncoding(nn.Module):
    def __init__(self, emb_dim: int, context_length: int, dtype=torch.float32):
        super(PositionEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.dtype = dtype
        self.position_encoding = self._compute_pos_encoding(seq_length=self.context_length)

    def _compute_pos_encoding(self, seq_length: int):
        if self.emb_dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(self.emb_dim))
        pe = torch.zeros(seq_length, self.emb_dim, dtype=self.dtype)
        position = torch.arange(0, seq_length, dtype=self.dtype).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.emb_dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / self.emb_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe

    def forward(self, input:torch.Tensor, seq_length: int | None = None)->torch.Tensor:
        if seq_length is not None:
            position_encoding = self._compute_pos_encoding(seq_length)
            B = input.shape[0]
            output = input + position_encoding.expand(B, -1, -1)
        else:
            B = input.shape[0]
            output = input+self.position_encoding.expand(B, -1, -1)
        return output


if(__name__ == "__main__"):
    

    tokens = 20
    dimensions = 256
    input = torch.zeros((tokens, dimensions))

    pos_encoding = PositionEncoding(dimensions, tokens)
    pe = pos_encoding(input)
    pe = pe[0].detach().cpu().numpy()
    print(pe.shape)

    plt.figure(figsize=(12,8))
    plt.pcolormesh(pe, cmap='viridis')
    plt.xlabel('Embedding Dimensions')
    plt.xlim((0, dimensions))
    plt.ylim((tokens,0))
    plt.ylabel('Token Position')
    plt.colorbar()
    plt.show()
    