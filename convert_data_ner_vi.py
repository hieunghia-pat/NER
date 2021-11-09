import json
from tqdm import tqdm
import re

train_data = open("phoner_data/word/train_word.conll").readlines()
val_data = open("phoner_data/word/dev_word.conll").readlines()
test_data = open("phoner_data/word/test_word.conll").readlines()

def create_sentences(data):
    sentences = []
    tags = []
    start_idx = 0
    for idx in tqdm(range(len(data))):
        try:
            data[idx] = data[idx].rstrip()
            data[idx] = data[idx].split()
            assert len(data[idx]) == 2, "Each sample must have 2 entities"
        except:
            sentences.append([sample[0] for sample in data[start_idx: idx]])
            tags.append([sample[1] for sample in data[start_idx: idx]])
            start_idx = idx+1

    refinded_data = []
    for sentence, tag in zip(sentences, tags):
        refinded_data.append({
            "sentence": sentence,
            "tag": tag
        })

    return refinded_data

train_data = create_sentences(train_data)
val_data = create_sentences(val_data)
test_data = create_sentences(test_data)

json.dump(train_data, open("phonert_train.json", "w+"), ensure_ascii=False)
json.dump(val_data, open("phonert_val.json", "w+"), ensure_ascii=False)
json.dump(test_data, open("phonert_test.json", "w+"), ensure_ascii=False)