# import
import argparse
import emoji
import re
import torch
import random
import os
import numpy as np
import multiprocessing
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from torch.utils.data import TensorDataset
from collections import defaultdict
from model.bert_cnn import BERT_CNN

# setup
print("Starting Hierarchical Classification...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.display.float_format = '{:,.2f}'.format  

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='init_data_repo_jtik.json', help='Dataset Path')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size')
parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased", help='BERT Model API')
parser.add_argument("--seed", type=int, default=42, help='Random Seed')
parser.add_argument("--max_epochs", type=int, default=30, help='Number of Epochs')
parser.add_argument("--lr", type=float, default=2e-5, help='Learning Rate')
parser.add_argument("--dropout", type=float, default=0.1, help='Dropout')
parser.add_argument("--patience", type=int, default=3, help='Patience')
parser.add_argument("--max_length", type=int, default=360, help='Max Length')
parser.add_argument("--in_channels", type=int, default=4, help='CNN In Channels')
parser.add_argument("--out_channels", type=int, default=32, help='CNN Out Channels')
parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5], help='CNN Kernel')
parser.add_argument("--num_bert_states", type=int, default=4, help='Number of BERT Last States')
config = vars(parser.parse_args())

np.random.seed(config["seed"]) 
torch.manual_seed(config["seed"])
random.seed(config["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

dataset = pd.read_json(f'dataset/{config["dataset"]}')
dataset = dataset[dataset['nama_pembimbing'] != '-']

stop_words = StopWordRemoverFactory().get_stop_words()
tokenizer = BertTokenizer.from_pretrained(config["bert_model"], use_fast=False)
stemmer = StemmerFactory().create_stemmer()
pretrained_bert = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=True)


# Generate Tree File
taxonomy = []

for column in ["prodi", "nama_pembimbing"]:
    print(column) 