# import
import argparse
import emoji
import re
import torch
import random
import os
import pickle
import numpy as np
import multiprocessing
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from torch.utils.data import TensorDataset


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='data_skripsi_jtik.csv')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--bert_model", type=str, default='indolem/indobert-base-uncased')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=3)
config = vars(parser.parse_args())

np.random.seed(config["seed"]) 
torch.manual_seed(config["seed"])
random.seed(config["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

dataset = pd.read_csv(config["dataset"])
stop_words = StopWordRemoverFactory().get_stop_words()
tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
stemmer = StemmerFactory().create_stemmer()
labels = dataset['prodi'].unique().tolist()
max_length = max(len(str(row["abstrak"]).split()) for row in dataset.tolist()) + 5


# preprocessor
if not os.path.exists("train_set.pkl") and not os.path.exists("valid_set.pkl") and not os.path.exists("test_set.pkl"):
    print("\nPreprocessing Data...")
    input_ids, attention_mask, target = [], [], []
    preprocessing_progress = tqdm(dataset.tolist()

    for row in preprocessing_progress:
        label = labels.index(row["prodi"])
        text = str(row["abstrak"]) 
        text = text.lower()
        text = emoji.replace_emoji(text, replace='') 
        text = re.sub(r'\n', ' ', text) 
        text = re.sub(r'http\S+', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^a-zA-Z ]', '', text)  
        text = ' '.join([word for word in text.split() if word not in stop_words])  
        text = stemmer.stem(text)
        text = text.strip()      

        token = tokenizer(text=text, max_length=max_length, padding="max_length", truncation=True)  
        input_ids.append(token['input_ids'])
        attention_mask.append(token['attention_mask'])
        target.append(label)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    target = torch.tensor(target)
    tensor_dataset = TensorDataset(input_ids, attention_mask, target)

    train_valid_size = round(len(tensor_dataset) * 0.8)
    test_size = len(tensor_dataset) - train_valid_size
    train_valid_set, test_set = torch.utils.data.random_split(tensor_dataset, [train_valid_size, test_size])

    train_size = round(len(train_valid_set) * 0.9)
    valid_size = len(train_valid_set) - train_size

    train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_size, valid_size])
    with open("train_set.pkl", 'wb') as train_preprocessed:
        pickle.dump(train_set, train_preprocessed)

    with open("valid_set.pkl", 'wb') as valid_preprocessed:
        pickle.dump(valid_set, valid_preprocessed)

    with open("test_set.pkl", 'wb') as test_preprocessed:
        pickle.dump(test_set, test_preprocessed)
    print('[ Preprocessing Completed ]\n')

print("\nLoading Data...")
with open("train_set.pkl", 'rb') as train_preprocessed:
    train_set = pickle.load(train_preprocessed)
    
with open("valid_set.pkl", 'rb') as valid_preprocessed:
    valid_set = pickle.load(valid_preprocessed)
    
with open("test_set.pkl", 'rb') as test_preprocessed:
    test_set = pickle.load(test_preprocessed)
print('[ Loading Completed ]\n')

train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                        batch_size=config["batch_size"], 
                                        shuffle=True,
                                        num_workers=multiprocessing.cpu_count())

valid_loader = torch.utils.data.DataLoader(dataset=valid_set, 
                                        batch_size=config["batch_size"], 
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                        batch_size=config["batch_size"], 
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())


# model
class BERT_CNN(nn.Module):
    def __init__(self, num_classes, bert_model, dropout, input_size=768, window_sizes=[1, 2, 3, 4, 5], in_channels=4, out_channels=32):
        super(BERT_CNN, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained(bert_model, output_hidden_states=True)

        conv_layers = []
        for window_size in window_sizes:
            conv_layer = nn.Conv2d(in_channels, out_channels, (window_size, input_size))
            conv_layers.append(conv_layer)
        self.cnn = nn.ModuleList(conv_layers)

        self.dropout = nn.Dropout(dropout) 
        self.window_length = len(window_sizes)
        self.out_channels_length = out_channels
        self.output_layer = nn.Linear(len(window_sizes) * out_channels, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_states = bert_output[2]
        bert_hidden_states = torch.stack(bert_hidden_states, dim=1)
        stacked_hidden_states = bert_hidden_states[:, -4:]

        pooling = []
        for layer in self.cnn:
            hidden_states = layer(stacked_hidden_states)
            relu_output = F.relu(hidden_states.squeeze(3))
            pooling.append(relu_output)

        max_pooling = []
        for features in pooling:
            pooled_features = F.max_pool1d(features, features.size(2)).squeeze(2)
            max_pooling.append(pooled_features)
        
        flatten = torch.cat(max_pooling, dim=1)
        logits = self.dropout(flatten)
        preds = self.output_layer(logits)
        
        return preds
    
model = BERT_CNN(len(labels), config["bert_model"], config["dropout"])
model.to(device)

n_total_steps = len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.9)
# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                         num_warmup_steps = 0,
#                                         num_training_steps = n_total_steps * config["max_epochs"])


# fine-tune
best_loss = 9.99
failed_counter = 0

print("Training Stage...")
for epoch in range(config["max_epochs"]):
    if failed_counter == config["patience"]:
        break

    model.train(True)
    for index, (input_ids, attention_mask, target) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(preds, target)

        loss.backward()
        optimizer.step()
        # scheduler.step()
        model.zero_grad()

        if (index+1) % config["batch_size"] == 0:
            print (f'Epoch [{epoch+1}/{config["max_epochs"]}], Step [{index+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        val_loss = 0
        n_samples = 0

        for input_ids, attention_mask, target in valid_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            preds = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(preds, target)
            val_loss += loss.item()
            n_samples += 1

            model.zero_grad()

        val_loss /= n_samples
        print(f'Epoch [{epoch+1}/{config["max_epochs"]}], Validation Loss: {val_loss:.4f}')
        
        if round(val_loss, 2) < round(best_loss, 2):
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            if os.path.exists('checkpoints/model_classifier.pt'):
                os.remove('checkpoints/model_classifier.pt')

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
                
            torch.save(checkpoint, 'checkpoints/model_classifier.pt')
            best_loss = val_loss
            failed_counter = 0

        else:
            failed_counter += 1

print("Test Stage...")
checkpoint = torch.load("checkpoints/model_classifier.pt")
print("Loading Checkpoint from Epoch", checkpoint['epoch'])
model.load_state_dict(checkpoint['model_state'])

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for input_ids, attention_mask, target in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)
        preds = model(input_ids=input_ids, attention_mask=attention_mask)

        result = torch.argmax(preds, dim=1) 
        n_samples += target.size(0)
        n_correct += (result == target).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Test Accuracy: {acc}%')
