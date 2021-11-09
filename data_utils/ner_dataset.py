import torch
from torch.utils.data import Dataset, DataLoader, random_split

from data_utils.vocab import Vocab
from data_utils.utils import collate_fn, preprocess_sentence

import json
import config

class NERDataset(Dataset):
    def __init__(self, json_dir, vocab=None):
        super(NERDataset, self).__init__()
        data = json.load(open(json_dir, encoding="latin1"))
        
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

        encoded_sen, seq_len = self.vocab._encode_sentence(sentence)
        encodede_tag = self.vocab._encode_tag(tag)

        return encoded_sen, encodede_tag, torch.tensor(seq_len)


    def get_kfolds(self, k=5):
        fold_size = int(len(self) * (1 / k))
        
        subdatasets = random_split(self, [fold_size]*(k-1) + [len(self) - fold_size*(k-1)], torch.Generator().manual_seed(13))

        folds = []
        for subdataset in subdatasets:
            folds.append(DataLoader(
                subdataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn
            ))

        return folds