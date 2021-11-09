import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics(object):
    def __init__(self, vocab=None) -> None:
        self.vocab = vocab

    def get_scores(self, predicted: torch.Tensor, true: torch.Tensor, avg_type="micro"):
        """ Compute the accuracies, precision, recall and F1 score for a batch of predictions and answers """

        predicted = self.vocab._decode_tag(predicted)
        true = self.vocab._decode_tag(true)

        acc = []
        pre = []
        recall = []
        f1 = []

        for y_hat, y in zip(predicted, true):
            acc.append(accuracy_score(true, predicted))
            pre.append(precision_score(true, predicted, average=avg_type))
            recall.append(recall_score(true, predicted, average=avg_type))
            f1.append(f1_score(true, predicted, average=avg_type))

        return {
            "accuracy": sum(acc) / len(acc),
            "precision": sum(pre) / len(pre),
            "recall": sum(recall) / len(recall),
            "F1": sum(f1) / len(f1)
        }