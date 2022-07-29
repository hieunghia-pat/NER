from torch.utils.data import Dataset

from data_utils.vocab import Vocab
from data_utils.utils import preprocess_sentence

import json

class NERDataset(Dataset):
    def __init__(self, json_dir, vocab=None):
        super(NERDataset, self).__init__()
        data = json.load(open(json_dir))
        
        self.sentences = []
        self.tags = []

        for sample in data:
            self.sentences.append(sample["sentence"])
            self.tags.append(sample["tag"])

        self.vocab = Vocab(data) if vocab is None else vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = preprocess_sentence(self.sentences[idx])
        tag = self.tags[idx]

        encoded_sen = self.vocab.encode_sentence(sentence)
        encoded_tag = self.vocab.encode_tag(tag)

        return encoded_sen, encoded_tag, len(sentence)