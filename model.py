import torch

from model_components import TransformerBlock, PositionEncoding

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, num_blocks: int, emb_dim: int, dim_ff: int, num_heads: int, context_length: int, dtype=torch.float32):
        super(Transformer, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoding = PositionEncoding(emb_dim, context_length)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(emb_dim, dim_ff, num_heads, context_length, dtype) for _ in range(num_blocks)])
        self.output_projection = torch.nn.Linear(in_features=emb_dim, out_features=vocab_size, bias=False)
    
    def forward(self, input: torch.Tensor)->torch.Tensor:
        x = self.embedding(input)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks: 
            x = block(x)
        x = self.output_projection(x)
        return torch.nn.functional.softmax(x, dim=-1) # final softmax-layer ??
