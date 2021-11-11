import json
import torch
from torch import nn
from torch.nn.modules import rnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.ner_self_attention import NERSelfAttention
from data_utils.vocab import Vocab
from data_utils.ner_dataset import NERDataset
from data_utils.utils import collate_fn
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
        # for macro evaluation
        macro_pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        macro_rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        macro_f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))
        # for micro evaluation
        micro_pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        micro_rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        micro_f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

        for s, t, _ in tq:
            s = s.to(device)
            t = t.to(device)
            
            out = model(s)
            macro_scores = metrics.get_scores(out.cpu(), t.cpu(), avg_type="macro")
            micro_scores = metrics.get_scores(out.cpu(), t.cpu(), avg_type="micro")

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
            acc_tracker.append(macro_scores["accuracy"])
            # macro evaluation
            macro_pre_tracker.append(macro_scores["precision"])
            macro_rec_tracker.append(macro_scores["recall"])
            macro_f1_tracker.append(macro_scores["F1"])
            # micro evaluation
            micro_pre_tracker.append(micro_scores["precision"])
            micro_rec_tracker.append(micro_scores["recall"])
            micro_f1_tracker.append(micro_scores["F1"])
            fmt = '{:.4f}:{:.4}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value, loss_tracker.mean.value), 
                            accuracy=fmt(acc_tracker.mean.value, acc_tracker.mean.value), 
                            precision=fmt(micro_pre_tracker.mean.value, macro_pre_tracker.mean.value), 
                            recall=fmt(micro_rec_tracker.mean.value, macro_rec_tracker.mean.value), 
                            f1=fmt(micro_f1_tracker.mean.value, macro_f1_tracker.mean.value))

        if not train:
            return {
                "accuracy": acc_tracker.mean.value,
                "precision": macro_pre_tracker.mean.value,
                "recall": macro_rec_tracker.mean.value,
                "F1": macro_f1_tracker.mean.value
            }

def main():

    train_data = json.load(open(config.json_file_train_vi))
    val_data = json.load(open(config.json_file_val_vi))
    test_data = json.load(open(config.json_file_test_vi))

    data = train_data + val_data + test_data

    vocab = Vocab(data, vectors=config.vectors, specials=["<pad>", "<s>", "</s>", "<unk>"], lower=False)

    train_dataset = NERDataset(config.json_file_train_vi, vocab=vocab, lower=False)
    val_dataset = NERDataset(config.json_file_val_vi, vocab=vocab, lower=False)
    test_dataset = NERDataset(config.json_file_test_vi, vocab=vocab, lower=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn= collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn= collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn= collate_fn
    )

    metrics.vocab = vocab
    loss_object = LabelSmoothingLoss(len(vocab.output_tags), vocab.stoi[vocab.pad], smoothing=config.smoothing).to(device)

    model = nn.DataParallel(NERSelfAttention(vocab=vocab, embedding_dim=config.embedding_dim, rnn_size=config.rnn_size, d_model=config.d_model,
                                                num_head=config.num_head, dff=config.dff, num_layers=config.num_layers).to(device))
    optimizer = Adam([p for p in model.parameters() if p.requires_grad])

    tracker = Tracker()

    max_f1 = 0 # for saving the best model
    f1_test = 0
    for e in range(config.epochs):
        run_epoch(model, [train_loader], loss_object, optimizer, tracker, train=True, prefix='Training', epoch=e)
        val_returned = run_epoch(model, [val_loader], loss_object, optimizer, tracker, train=False, prefix='Validation', epoch=e)
        test_returned = run_epoch(model, [test_loader], loss_object, optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

        print("+"*13)

        results = {
            'weights': model.state_dict(),
            'val': {
                'accuracy': val_returned["accuracy"],
                "precision": val_returned["precision"],
                "recall": val_returned["recall"],
                "f1": val_returned["F1"],

            },
            'test': {
                'accuracy': test_returned["accuracy"],
                "precision": test_returned["precision"],
                "recall": test_returned["recall"],
                "f1": test_returned["F1"],

            },
            'vocab': vocab,
        }
    
        torch.save(results, os.path.join(config.model_checkpoint, f"model_vi_last.pth"))
        if val_returned["F1"] > max_f1:
            max_f1 = val_returned["F1"]
            f1_test = test_returned["F1"]
            torch.save(results, os.path.join(config.model_checkpoint, f"model_vi_best.pth"))

    print(f"Training finished. Best F1 score on dev set: {max_f1}. Score on test set: {f1_test}")
    print("="*31)        

if __name__ == "__main__":
    main()