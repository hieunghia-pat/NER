from torch import nn

from model.embedding import PretrainedEmbeddings

class NERSelfAttention(nn.Module):
    def __init__(self, vocab, embedding_dim, rnn_size, d_model, num_head, dff, num_layers, dropout=0.5) -> None:
        super(NERSelfAttention, self).__init__()

        self.embedding = PretrainedEmbeddings(embedding_dim, vocab, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                d_model=d_model, 
                                                nhead=num_head,
                                                dim_feedforward=dff,
                                                batch_first=True,
                                                dropout=dropout), num_layers=num_layers)
        self.fc = nn.Linear(rnn_size*2, len(vocab.output_tags))

    def forward(self, s):
        embedded = self.dropout(self.embedding(s))
        feature = self.encoder(embedded)
        out = self.fc(feature)

        return out