import torch
from torch import nn
from torch.optim import Adam

from model.ner_bilstm import NERBiLSTM
from data_utils.ner_dataset import NERDataset
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker
from loss_utils.label_smoothing_loss import LabelSmoothingLoss

import config
from tqdm import tqdm
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
metrics = Metrics()

total_iterations = 0

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_epoch(model, loaders, loss_func, optimizer, tracker, train=False, prefix="", epoch=0):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    for loader in loaders:
        tq = tqdm(loader, desc='Epoch {:03d} - {} - Fold {}'.format(epoch, prefix, loaders.index(loader)+1), ncols=0)
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
        acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
        pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

        for s, t, s_len in tq:
            s = s.to(device)
            t = t.to(device)
            s_len = s_len.to(device)

            out = model(s, t, s_len)
            scores = metrics.get_scores(out.cpu(), out.cpu())

            if train:
                global total_iterations
                update_learning_rate(optimizer, total_iterations)

                optimizer.zero_grad()
                loss = loss_func(out, t)
                loss.backward()
                optimizer.step()

                total_iterations += 1
            else:
                loss = np.array(0)

            loss_tracker.append(loss.item())
            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), accuracy=fmt(acc_tracker.mean.value), 
                            precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))

        if not train:
            return {
                "accuracy": acc_tracker.mean.value,
                "precision": pre_tracker.mean.value,
                "recall": rec_tracker.mean.value,
                "F1": f1_tracker.mean.value
            }

def main():
    
    dataset = NERDataset(config.json_file)
    k_fold = 5
    folds = dataset.get_kfolds(k_fold)
    metrics.vocab = dataset.vocab
    loss_object = LabelSmoothingLoss(len(dataset.vocab.output_tags), dataset.vocab.stoi["<pad>"], smoothing=config.smoothing).to(device)

    for k in range(k_fold):
        print(f"Stage {k+1}:")
        model = nn.DataParallel(NERBiLSTM(len(dataset.vocab), config.embedding_dim, config.rnn_size, len(dataset.vocab.output_tags)).to(device))
        optimizer = Adam([p for p in model.parameters() if p.requires_grad])

        tracker = Tracker()

        max_f1 = 0 # for saving the best model
        for e in range(config.epochs):
            run_epoch(model, folds[:-1], loss_object, optimizer, tracker, train=True, prefix='Training', epoch=e)
            val_returned = run_epoch(model, [folds[-1]], loss_object, optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

            print("+"*13)

            results = {
                'weights': model.state_dict(),
                'eval': {
                    'accuracy': val_returned["accuracy"],
                    "precision": val_returned["precision"],
                    "recall": val_returned["recall"],
                    "f1": val_returned["F1"],

                },
                'vocab': dataset.vocab,
            }
        
            torch.save(results, os.path.join(config.model_checkpoint, f"model_last_stage_{k+1}.pth"))
            if val_returned["F1"] > max_f1:
                max_f1 = val_returned["F1"]
                torch.save(results, os.path.join(config.model_checkpoint, f"model_best_stage_{k+1}.pth"))

        print(f"Finished for stage {k+1}. Best F1 score: {max_f1}.")
        print("="*31)

        # change roles of the folds
        for i in range(k_fold):
            tmp = folds[i]
            folds[i] = folds[i-1]
            folds[i-1] = tmp
        
        
if __name__ == "__main__":
    main()