import pandas as pd
import json
from tqdm import tqdm

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

isSentences_att = data["Sentence #"].tolist()
words_att = data["Word"].tolist()
pos_att = data["POS"].tolist()
tags_att = data["Tag"].tolist()

assert len(isSentences_att) == len(words_att) == len(pos_att) == len(tags_att), "Length of the three attributes must be equal"
sentences = []
total_sentences = list(set(isSentences_att))
total_sentences.sort()

for sentence_id in tqdm(total_sentences):
    ids = [id for id in range(len(isSentences_att)) if isSentences_att[id] == sentence_id]
    sentence = [words_att[id] for id in ids]
    pos = [pos_att[id] for id in ids]
    tag = [tags_att[id] for id in ids]
    sentences.append({
        "sentence": sentence,
        "pos": pos,
        "tag": tag
    })


json.dump(sentences, open("ner_dataset.json", "w+", encoding="latin1"), ensure_ascii="False")