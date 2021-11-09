from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from model.embedding import Embeddings

class NERBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_size, num_tags, dropout=0.5) -> None:
        super(NERBiLSTM, self).__init__()

        self.embedding = Embeddings(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_size,  batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_size*2, num_tags)

    def forward(self, s, s_len):
        embedded = self.dropout(self.embedding(s))
        # packed = pack_padded_sequence(embedded, s_len.cpu(), batch_first=True)
        out, _ = self.rnn(embedded)
        out = self.fc(out)

        return out