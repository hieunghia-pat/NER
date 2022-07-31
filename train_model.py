import torch
import torch.nn as nn
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp

from data_utils.vocab import Vocab
from data_utils.ner_dataset import NERDataset
from torch.optim import Adam
from utils import get_model, get_config
from evaluation import compute_scores

import os
from tqdm import tqdm
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model: nn.Module, epoch: int, dataloader: data.DataLoader, loss_fn, optim):
    vocab = dataloader.dataset.vocab
    model.train()
    running_loss = 0.
    with tqdm(desc=f"Epoch {epoch} - Training", unit="it", total=len(dataloader)) as pbar:
        for ith, (sentences, tags, sentence_lens) in enumerate(dataloader):
            sentences = sentences.to(device)
            tags = tags.to(device)
            outs = model(sentences)

            optim.zero_grad()
            loss = loss_fn(outs.view(-1, len(vocab.tags)), tags.view(-1))
            loss.backward()
            
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (ith + 1))
            pbar.update()

def evaluate_loss(model: nn.Module, epoch: int, dataloader: data.DataLoader, loss_fn):
    vocab = dataloader.dataset.vocab
    model.eval()
    running_loss = 0.
    with tqdm(desc=f"Epoch {epoch} - Evaluating", unit="it", total=len(dataloader)) as pbar:
        for ith, (sentences, tags, sentence_lens) in enumerate(dataloader):
            sentences = sentences.to(device)
            tags = tags.to(device)
            outs = model(sentences)
            
            loss = loss_fn(outs.view(-1, len(vocab.tags)), tags.view(-1))
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (ith + 1))
            pbar.update()

def evaluate_metrics(model: nn.Module, epoch: int, dataloader: data.DataLoader):
    vocab = dataloader.dataset.vocab
    model.eval()
    predicteds = []
    gts = []
    lens = []
    with tqdm(desc=f"Epoch {epoch} - Evaluating", unit="it", total=len(dataloader)) as pbar:
        for ith, (sentences, tags, sentence_lens) in enumerate(dataloader):
            sentences = sentences.to(device)
            tags = tags.to(device)
            predicted_tags = model(sentences).argmax(dim=-1)
            gt_tags = vocab.decode_tag(gt_tags, sentence_lens)
            predicted_tags = vocab.decode_tag(predicted_tags, sentence_lens)
            
            predicteds += predicted_tags
            gts = gt_tags
            lens += sentence_lens.tolist()
            pbar.update()
        
    scores = compute_scores(predicteds, gts, sentence_lens)
    return scores

def main(processor, configs):
    rank = configs.rank * configs.process_per_node + processor
    print('initializing...')
    dist.init_process_group(backend='gloo', init_method='env://', world_size=configs.world_size, rank=rank)
    torch.manual_seed(13)
    
    if not os.path.isdir(os.path.join(configs.checkpoint, configs.pretrained_language_model_name)):
        print("Creating the checkpoint path ...")
        os.makedirs(os.path.join(configs.checkpoint, configs.pretrained_language_model_name))
    
    if not os.path.isfile(os.path.join(configs.checkpoint, configs.pretrained_language_model_name, "vocab.pkl")):
        print("Defining vocab ...")
        vocab = Vocab([
            configs.train_json_dir,
            configs.val_json_dir,
            configs.test_json_dir
        ])
        pickle.dump(vocab, open(os.path.join(configs.checkpoint, configs.pretrained_language_model_name, "vocab.pkl"), "wb"))
    else:
        print("Loading vocab ...")
        vocab = pickle.load(open(os.path.join(configs.checkpoint, configs.pretrained_language_model_name, "vocab.pkl"), "rb"))

    print("Create datasets ...")
    train_dataset = NERDataset(configs.train_json_dir, vocab=vocab)
    val_dataset = NERDataset(configs.val_json_dir, vocab=vocab)
    test_dataset = NERDataset(configs.test_json_dir, vocab=vocab)

    print("Creating data loaders ...")
    sampler = data.distributed.DistributedSampler(train_dataset,
                                                    num_replicas=configs.world_size,
                                                    rank=rank)

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.workers,
        sampler=sampler
    )
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.workers,
        shuffle=False
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.workers,
        shuffle=False
    )

    print("Creating the model ...")
    model = get_model(configs, vocab).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    print("Defining loss and optimizer ...")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=configs.label_smoothing)
    optimizer = Adam(model.parameters(), lr=configs.learning_rate)

    epoch = 0
    pantient = 0
    best_f1 = 0
    while True:
        epoch += 1

        train(model, epoch, train_dataloader, loss_fn, optimizer)
        
        if configs.rank == 0:
            evaluate_loss(model, epoch, val_dataloader, loss_fn)

            scores = evaluate_metrics(model, epoch, val_dataloader)
            print(f"Validation scores: {scores}")
            f1 = scores["f1"]
            if best_f1 < f1:
                best_f1 = f1

            scores = evaluate_metrics(model, epoch, test_dataloader)
            print(f"Evaluation scores: {scores}")

            if f1 < best_f1:
                pantient += 1

            torch.save(model.state_dict(), os.path.join(configs.checkpoint, configs.pretrained_language_model_name, "last_model.pth"))
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(configs.checkpoint, configs.pretrained_language_model_name, "best_model.pth"))

            if pantient == 5:
                break

if __name__ == "__main__":
    configs = get_config("config.yaml")
    configs.world_size = configs.process_per_node * configs.nodes
    os.environ['MASTER_ADDR'] = configs.root_address
    os.environ['MASTER_PORT'] = configs.root_port
    mp.spawn(main, nprocs=configs.process_per_node, args=(configs, ))