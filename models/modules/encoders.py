import torch
from torch import nn
from torch.nn import functional as F

from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.attentions import MultiHeadAttention, ScaledDotProductAttention
from models.utils import generate_padding_mask
from models.modules.embeddings import SinusoidPositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 use_aoa=False, attention_module=None, **attention_module_kwargs):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        use_aoa=use_aoa,
                                        attention_module=attention_module,
                                        **attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)
        ff = ff.masked_fill(attention_mask.squeeze().unsqueeze(-1), value=0)

        return ff

class Encoder(nn.Module):
    def __init__(self, N, padding_idx, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, multi_level_output=False,
                 identity_map_reordering=False, use_aoa=False, **attention_module_kwargs):
        super(Encoder, self).__init__()

        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.pos_embedding = SinusoidPositionalEmbedding(d_model, normalize=True)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  use_aoa=use_aoa,
                                                  attention_module=ScaledDotProductAttention,
                                                  **attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.multi_level_output = multi_level_output

    def forward(self, features, **kwargs):
        padding_masks = generate_padding_mask(features, padding_idx=0).unsqueeze(1).unsqueeze(1) # (bs, 1, 1, seq_len)

        features = F.relu(self.fc(features))
        features = self.dropout(features)
        out = self.layer_norm(features)

        if self.multi_level_output:
            outs = []
        pos_embedding = self.pos_embedding(out)
        for layer in self.layers:
            out = out + pos_embedding
            out = layer(queries=out, keys=out, values=out, attention_mask=padding_masks)
            if self.multi_level_output:
                outs.append(out.unsqueeze(1))

        if self.multi_level_output:
            outs = torch.cat(outs, dim=1)
            return outs, padding_masks, pos_embedding
        
        return out, padding_masks, pos_embedding