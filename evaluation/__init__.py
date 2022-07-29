import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_utils.vocab import Vocab

def compute_scores(predicteds: List[List[int]], gts: List[List[int]], sentence_lens: List[int]):
    acc = []
    pre = []
    recall = []
    f1 = []
    for predicted, gt, sentence_len in zip(predicteds, gts, sentence_lens):
        predicted = predicted[:sentence_len]
        gt = gt[:sentence_len]
        acc.append(accuracy_score(gt, predicted))
        pre.append(precision_score(gt, predicted, average="macro", zero_division=0))
        recall.append(recall_score(gt, predicted, average="macro", zero_division=0))
        f1.append(f1_score(gt, predicted, average="macro"))

    return {
        "accuracy": np.array(acc).mean(),
        "precision": np.array(pre).mean(),
        "recall": np.array(recall).mean(),
        "f1": np.array(f1).mean()
    }

def compute_scores_per_class(vocab: Vocab, predicteds: List[List[int]], gts: List[List[int]], sentence_lens: List[int]):
    tags = set()
    for tag in vocab.tags:
        if tag == "O":
            continue
        tags.add(tag.split("-")[-1])
    tags = list(tags)
    scores_per_class = []
    for tag in tags:
        scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        for predicted, gt, sentence_len in zip(predicteds, gts, sentence_lens):
            predicted = predicted[:sentence_len]
            gt = gt[:sentence_len]
            refined_predicted = []
            refined_gt = []
            for predicted_tag, gt_tag in zip(predicted, gt):
                if predicted_tag == tag or gt_tag == tag:
                    refined_predicted.append(predicted_tag)
                    refined_gt.append(gt_tag)
            scores["accuracy"].append(accuracy_score(refined_gt, refined_predicted) if len(refined_gt) > 0 else 0)
            scores["precision"].append(precision_score(refined_gt, refined_predicted, average="macro", zero_division=0) if len(refined_gt) > 0 else 0)
            scores["recall"].append(recall_score(refined_gt, refined_predicted, average="macro", zero_division=0) if len(refined_gt) > 0 else 0)
            scores["f1"].append(f1_score(refined_gt, refined_predicted, average="macro") if len(refined_gt) > 0 else 0)

        scores["accuracy"] = np.array(scores["accuracy"]).mean()
        scores["precision"] = np.array(scores["precision"]).mean()
        scores["recall"] = np.array(scores["recall"]).mean()
        scores["f1"] = np.array(scores["f1"]).mean()

        scores_per_class.append(scores)

    return list(zip(tags, scores_per_class))