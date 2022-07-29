import torch
from torch import nn
from transformers import GPT2Model
from models.modules.attentions import ScaledDotProductAttention

from models.modules.encoders import EncoderLayer
from models.modules.embeddings import Embedding, SinusoidPositionalEmbedding
from data_utils.vocab import Vocab
from models.utils import generate_padding_mask

class GPTModel(nn.Module):
    def __init__(self, vocab: Vocab, pretrained_language_model_name: str, pretrained_language_model_dim=768,
                    embedding_dim=300, d_model=512, dropout=0.5):
        super(GPTModel, self).__init__()

        self.padding_idx = vocab.padding_idx
        
        self.embedding = Embedding(len(vocab), embedding_dim, weights=vocab.vectors, padding_idx=vocab.padding_idx)
        self.pos_embedding = SinusoidPositionalEmbedding(embedding_dim, normalize=True)

        self.proj_to_language_model = nn.Linear(embedding_dim, pretrained_language_model_dim)
        
        language_model = GPT2Model.from_pretrained(pretrained_language_model_name)
        self.language_model_encoder = language_model.h
        self.language_model_norm = language_model.ln_f

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
        outs = self.language_model_encoder(outs, attention_mask=torch.logical_not(padding_mask)).last_hidden_state
        outs = self.language_model_norm(outs)
        # project to d_model
        outs = self.layer_norm(self.proj_to_model(outs))
        # fine-tuning the pretrained language model
        outs = self.encoder(queries=outs, keys=outs, values=outs, attention_mask=padding_mask)
        # project to NER tags
        outs = self.dropout(self.fc(outs))

        return outs