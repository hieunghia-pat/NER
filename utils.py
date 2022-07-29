from models import *

from yacs.config import CfgNode
import yaml

def get_config(yaml_file):
    return CfgNode(init_dict=yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader))

models = {
    "bert-base-multilingual-cased": BERTModel,
    "bert-base-multilingual-uncased": BERTModel,
    "bert-large-multilingual-cased": BERTModel,
    "bert-large-multilingual-uncased": BERTModel,
    "vinai/phobert-base": RoBERTaModel,
    "vinai/phobert-large": RoBERTaModel,
    "vinai/bartpho-syllable": BARTModel,
    "vinai/bartpho-word": BARTModel,
    "VietAI/vit5-base": T5Model,
    "VietAI/vit5-large": T5Model,
    "google/mt5-base": T5Model,
    "google/mt5-large": T5Model
}

model_names = {
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased": "bert-base-multilingual-uncased",
    "bert-large-multilingual-cased": "bert-large-multilingual-cased",
    "bert-large-multilingual-uncased": "bert-large-multilingual-uncased",
    "phobert-base": "vinai/phobert-base",
    "phobert-large": "vinai/phobert-large",
    "bartpho-syllable": "vinai/bartpho-syllable",
    "bartpho-word": "vinai/bartpho-word",
    "vit5-base": "VietAI/vit5-base",
    "vit5-large": "VietAI/vit5-large",
    "mt5-base": "google/mt5-base",
    "mt5-large": "google/mt5-large"
}

def get_model(config, vocab):
    model_name = model_names[config.pretrained_language_model_name]
    model = models[model_name]
    return model(vocab=vocab, pretrained_language_model_name=model_name, 
                    pretrained_language_model_dim=config.pretrained_language_model_dim,
                    embedding_dim=config.embedding_dim, d_model=config.d_model, dropout=config.dropout)