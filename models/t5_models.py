import torch
from torch import nn
from transformers import T5Model as T5_model
from models.modules.attentions import ScaledDotProductAttention

from models.modules.encoders import EncoderLayer
from models.modules.embeddings import Embedding, SinusoidPositionalEmbedding
from data_utils.vocab import Vocab
from models.utils import generate_padding_mask

class T5Model(nn.Module):
    def __init__(self, vocab: Vocab, pretrained_language_model_name: str, pretrained_language_model_dim=768,
                    embedding_dim=300, d_model=512, dropout=0.5):
        super(T5Model, self).__init__()

        self.padding_idx = vocab.padding_idx
        
        self.embedding = Embedding(len(vocab), embedding_dim, weights=vocab.vectors, padding_idx=vocab.padding_idx)
        self.pos_embedding = SinusoidPositionalEmbedding(embedding_dim, normalize=True)

        self.proj_to_language_model = nn.Linear(embedding_dim, pretrained_language_model_dim)
        
        language_model = T5_model.from_pretrained(pretrained_language_model_name)
        self.language_model_encoder_blocks = language_model.encoder.block
        self.language_model_norm = language_model.encoder.final_layer_norm
        self.language_model_dropout = language_model.encoder.dropout

        self.proj_to_model = nn.Linear(pretrained_language_model_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = EncoderLayer(d_model, attention_module=ScaledDotProductAttention)
        
        self.fc = nn.Linear(d_model, len(vocab.tags))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, tokens):
        padding_mask = generate_padding_mask(tokens, padding_idx=self.padding_idx).unsqueeze(1).unsqueeze(1)
        
        outs = self.embedding(tokens) + self.pos_embedding(tokens)
        outs = self.proj_to_language_model(outs)
        # feed to the pretrained language model
        for block in self.language_model_encoder_blocks:
            outs = block(outs, attention_mask=torch.logical_not(padding_mask))[0]
        outs = self.language_model_norm(outs)
        outs = self.language_model_dropout(outs)
        # project to d_model
        outs = self.layer_norm(self.proj_to_model(outs))
        # fine-tuning the pretrained language model
        outs = self.encoder(queries=outs, keys=outs, values=outs, attention_mask=padding_mask)
        # project to NER tags
        outs = self.dropout(self.fc(outs))

        return outs