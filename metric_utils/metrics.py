import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict

class Metrics(object):
    def __init__(self, vocab=None) -> None:
        self.vocab = vocab

    def get_scores(self, predicted: torch.Tensor, true: torch.Tensor, avg_type="micro") -> Dict(str, float):
        """ Compute the accuracies, precision, recall and F1 score for a batch of predictions and answers """

        predicted = self.vocab._decode_tag(predicted)
        true = self.vocab._decode_tag(true)

        acc = accuracy_score(true, predicted)
        pre = precision_score(true, predicted, average=avg_type)
        recall = recall_score(true, predicted, average=avg_type)
        f1 = f1_score(true, predicted, average=avg_type)

        return {
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "F1": f1
        }