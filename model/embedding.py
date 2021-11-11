import torch
from torch import nn
import math

class Embeddings(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embedding_dim)

class PretrainedEmbeddings(Embeddings):
    def __init__(self, embedding_dim, vocab, d_model=None):
        super(PretrainedEmbeddings, self).__init__(embedding_dim, len(vocab.itos))
        if vocab.vectors is not None:
            assert d_model is not None, "d_model must be specified together with vocab"
            self.lut.from_pretrained(vocab.vectors)
            self.fc = nn.Linear(embedding_dim, d_model)
        else:
            self.fc = lambda x: x
        
    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.embedding_dim)
        
        return self.fc(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)