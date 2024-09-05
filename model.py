import torch

from model_components import TransformerBlock, PositionEncoding

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, num_blocks: int, emb_dim: int, dim_ff: int, num_heads: int, context_length: int, dtype=torch.float32):
        super(Transformer, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoding = PositionEncoding(emb_dim, context_length)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(emb_dim, dim_ff, num_heads, context_length, dtype) for _ in range(num_blocks)])
        self.output_projection = torch.nn.Linear(in_features=emb_dim, out_features=vocab_size, bias=False)
    
    def forward(self, input: torch.Tensor, seq_length: int | None = None)->torch.Tensor:
        x = self.embedding(input)
        x = self.pos_encoding(x, seq_length)
        for block in self.transformer_blocks: 
            x = block(x)
        x = self.output_projection(x)
        return x #torch.nn.functional.softmax(x, dim=-1) # final softmax-layer ??
    
    def _draw_sample(self, next_token: torch.Tensor):
        weights = torch.exp(next_token)
        sample = torch.multinomial(weights, num_samples=1)
        #print("Argmax sample: ", torch.argmax(sample, dim=1))
        return sample # [B, 1]

    
    def sample_sequence(self, start_seq: torch.Tensor, length_sequence: int)->list[int]:
        start_seq = start_seq.expand(1, -1) # [B, 1]
        print("Start seq: ", start_seq.shape)
        for _ in range(length_sequence):
            seq_length = start_seq.shape[1]
            x = self.forward(start_seq, seq_length=seq_length)
            next_token = x[:, -1, :]
            # draw categorical sample
            sample = self._draw_sample(next_token)
            # concatenate
            start_seq = torch.concat([start_seq, sample], dim=1)

        return start_seq

