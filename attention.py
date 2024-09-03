import torch

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self,dim_in: int, dim_out: int, dtype=torch.float32):
        super(SelfAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dtype = dtype
        self.keys_layer = torch.nn.Linear(dim_in, dim_out, bias = False, dtype=self.dtype)
        self.queries_layer = torch.nn.Linear(dim_in, dim_out, bias = False, dtype=self.dtype)
        self.values_layer = torch.nn.Linear(dim_in, dim_out, bias=False, dtype=self.dtype)
        self.softmax_layer = torch.nn.Softmax(dim=-1)
    def get_att_scores(self, keys, queries):
        """
        Implements computation of attention scores:
        :params keys: keys [B, N, D]
        :params queries: [B, N, D]
        """
        return self.softmax_layer(torch.einsum('bid, bjd->bij', queries, keys)/(self.dim_out**0.5))
    def forward(self, x):
        """
        Implements forward pass of self-attention layer
        params:
        x: [B, N, D] (B: batch size, N: context-length, D: latent space dim.)
        """
        keys = self.keys_layer(x)
        queries = self.queries_layer(x)
        values = self.values_layer(x)
        # attention mechanism
        att_scores = self.get_att_scores(keys, queries) # shape: [B, N, N]

        return torch.einsum('bjd, bij->bid', values, att_scores) # shape: [B, N, D]


class MaskedSelfAttentionLayer(torch.nn.Module):
    def __init__(self,dim_in: int, dim_out: int, dtype=torch.float32):
        super(MaskedSelfAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dtype = dtype
        self.keys_layer = torch.nn.Linear(dim_in, dim_out, bias = False, dtype=self.dtype)
        self.queries_layer = torch.nn.Linear(dim_in, dim_out, bias = False, dtype=self.dtype)
        self.values_layer = torch.nn.Linear(dim_in, dim_out, bias=False, dtype=self.dtype)
        self.softmax_layer = torch.nn.Softmax(dim=-1)
    def _get_att_scores(self, keys, queries):
        """
        Implements computation of attention scores:
        :params keys: keys [B, N, D]
        :params queries: [B, N, D]
        """
        B, N, _ = keys.shape
        att_mask = torch.tril(torch.ones(N,N)).expand(B, -1, -1)
        unmasked_att = torch.einsum('bid, bjd->bij', queries, keys)/(self.dim_out**0.5)

        # apply masking
        att = unmasked_att.masked_fill(att_mask[:,:N, :N] == 0, float('-inf')) # [B, N, N]

        return self.softmax_layer(att)
    def forward(self, x):
        """
        Implements forward pass of self-attention layer
        params:
        x: [B, N, D] (B: batch size, N: context-length, D: latent space dim.)
        """
        keys = self.keys_layer(x)
        queries = self.queries_layer(x)
        values = self.values_layer(x)
        # attention mechanism
        att_scores = self._get_att_scores(keys, queries) # shape: [B, N, N]

        return torch.einsum('bjd, bij->bid', values, att_scores) # shape: [B, N, D]

        
class MultiHeadSelfAttentionLayer(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, context_length: int, dtype=torch.float32):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.d_head = dim_out // num_heads
        self.context_length = context_length
        self.dtype = dtype
        self.output_projection = torch.nn.Linear(dim_out, dim_out, dtype=self.dtype)
        self.attention_heads = torch.nn.ModuleList(
            [MaskedSelfAttentionLayer(dim_in, self.d_head, dtype=self.dtype) for _ in range(self.num_heads)]
        )
    def forward(self, x):
        x = torch.cat([self_att(x) for self_att in self.attention_heads], dim=-1) # concatenate along data dim
        return self.output_projection(x)