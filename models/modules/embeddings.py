import torch
from torch import nn

import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_emb, weights=None, padding_idx=0):
        super(Embedding, self).__init__()
        if weights is None:
            self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx)
        else:
            assert d_emb == weights.shape[-1]
            self.embedding = nn.Embedding.from_pretrained(embeddings=weights, freeze=True, padding_idx=padding_idx)

    def forward(self, tokens):
        return self.embedding(tokens)

class SinusoidPositionalEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # returns an 1D embedding for visual feature
    # follow implementation of https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/position_encoding.py

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(SinusoidPositionalEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=-1).flatten(-2)

        return pos