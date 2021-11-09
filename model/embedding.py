from torch import nn
import math


class Embeddings(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embedding_dim)