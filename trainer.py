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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.display.float_format = '{:,.2f}'.format  

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='data_repo_jtik.csv')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--max_length", type=int, default=360)
parser.add_argument("--in_channels", type=int, default=4)
parser.add_argument("--out_channels", type=int, default=32)
parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
parser.add_argument("--num_bert_states", type=int, default=4)
config = vars(parser.parse_args())

np.random.seed(config["seed"]) 
torch.manual_seed(config["seed"])
random.seed(config["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

dataset = pd.read_csv(f'dataset/{config["dataset"]}')
stop_words = StopWordRemoverFactory().get_stop_words()
tokenizer = BertTokenizer.from_pretrained(config["bert_model"], use_fast=False)
stemmer = StemmerFactory().create_stemmer()
labels = sorted(dataset['prodi'].unique().tolist())
pretrained_bert = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=True)


# preprocessor
if not os.path.exists("dataset/preprocessed/flat_train_set.pt") and not os.path.exists("dataset/preprocessed/flat_valid_set.pt") and not os.path.exists("dataset/preprocessed/flat_test_set.pt"):
    print("\nPreprocessing Data...")
    input_ids, attention_mask, target = [], [], []

    for row in tqdm(dataset.to_dict('records'), desc="Preprocessing"):
        label = labels.index(row["prodi"])
        text = str(row["kata_kunci"]) + " - " + str(row["abstrak"])
        text = text.lower()
        text = emoji.replace_emoji(text, replace='') 
        text = re.sub(r'\n', ' ', text) 
        text = re.sub(r'http\S+', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^a-zA-Z ]', '', text)  
        text = ' '.join([word for word in text.split() if word not in stop_words])  
        text = stemmer.stem(text)
        text = text.strip()      

        token = tokenizer(text=text, max_length=config["max_length"], padding="max_length", truncation=True)  
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

    if not os.path.exists('dataset/preprocessed'):
        os.makedirs('dataset/preprocessed')

    train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_size, valid_size])
    torch.save(train_set, 'dataset/preprocessed/flat_train_set.pt')
    torch.save(valid_set, 'dataset/preprocessed/flat_valid_set.pt')
    torch.save(test_set, 'dataset/preprocessed/flat_test_set.pt')
    print('[ Preprocessing Completed ]\n')

print("\nLoading Data...")
train_set = torch.load("dataset/preprocessed/flat_train_set.pt")
valid_set = torch.load("dataset/preprocessed/flat_valid_set.pt")
test_set = torch.load("dataset/preprocessed/flat_test_set.pt")
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


# fine-tune
model = BERT_CNN(pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
model.to(device)

output_layer = nn.Linear(len(config["window_sizes"]) * config["out_channels"], len(labels))
output_layer.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

best_loss = 9.99
failed_counter = 0

logger = pd.DataFrame(columns=['accuracy', 'loss', 'epoch', 'stage']) 
classification_report = pd.DataFrame(columns=['label', 'correct_prediction', 'false_prediction', 'total_prediction', 'epoch', 'stage'])

optimizer.zero_grad()
model.zero_grad()
output_layer.zero_grad()

print("Training Stage...")
for epoch in range(config["max_epochs"]):
    if failed_counter == config["patience"]:
        print("Early Stopping")
        break

    train_loss = 0
    n_batch = 0
    n_correct = 0
    n_samples = 0

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    model.train(True)
    for input_ids, attention_mask, target in tqdm(train_loader, desc="Training Stage", unit="batch"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = output_layer(preds)

        loss = criterion(preds, target)

        train_loss += loss.item()
        n_batch += 1

        result = torch.argmax(preds, dim=1) 
        n_correct += (result == target).sum().item()
        n_samples += target.size(0)

        for prediction, ground_truth in zip(result, target):
            if prediction == ground_truth:
                each_label_correct[ground_truth.item()] += 1
            each_label_total[ground_truth.item()] += 1

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        model.zero_grad()
        output_layer.zero_grad()

    train_loss /= n_batch
    acc = 100.0 * n_correct / n_samples
    logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [train_loss], 'epoch': [epoch+1], 'stage': ['train']})], ignore_index=True)
    print(f'Epoch [{epoch + 1}/{config["max_epochs"]}], Training Loss: {train_loss:.4f}, Training Accuracy: {acc:.2f}%')

    for label, total_count in each_label_total.items():
        correct_count = each_label_correct.get(label, 0)  
        false_count = total_count - correct_count
        classification_report = pd.concat([classification_report, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['train']})], ignore_index=True)
        print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

    model.eval()
    with torch.no_grad():
        val_loss = 0
        n_batch = 0
        n_correct = 0
        n_samples = 0

        each_label_correct = defaultdict(int)
        each_label_total = defaultdict(int)

        for input_ids, attention_mask, target in tqdm(valid_loader, desc="Validation Stage", unit="batch"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = output_layer(preds)

            loss = criterion(preds, target)

            val_loss += loss.item()
            n_batch += 1

            result = torch.argmax(preds, dim=1) 
            n_correct += (result == target).sum().item()
            n_samples += target.size(0)

            for prediction, ground_truth in zip(result, target):
                if prediction == ground_truth:
                    each_label_correct[ground_truth.item()] += 1
                each_label_total[ground_truth.item()] += 1

            optimizer.zero_grad()
            model.zero_grad()
            output_layer.zero_grad()

        val_loss /= n_batch
        acc = 100.0 * n_correct / n_samples
        logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [val_loss], 'epoch': [epoch+1], 'stage': ['valid']})], ignore_index=True)
        print(f'Epoch [{epoch + 1}/{config["max_epochs"]}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc:.2f}%')

        for label, total_count in each_label_total.items():
            correct_count = each_label_correct.get(label, 0)  
            false_count = total_count - correct_count
            classification_report = pd.concat([classification_report, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['valid']})], ignore_index=True)
            print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")
        
        if round(val_loss, 2) < round(best_loss, 2):
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            if os.path.exists('checkpoints/flat_model.pt'):
                os.remove('checkpoints/flat_model.pt')

            print("Saving Checkpoint...")

            checkpoint = {
                "hidden_states": model.state_dict(),
                "last_hidden_state": output_layer.state_dict(),
            }

            torch.save(checkpoint, 'checkpoints/flat_model.pt')

            best_loss = val_loss
            failed_counter = 0

        else:
            failed_counter += 1

print("Test Stage...")
checkpoint = torch.load('checkpoints/flat_model.pt', map_location=device)
model.load_state_dict(checkpoint["hidden_states"])
output_layer.load_state_dict(checkpoint["last_hidden_state"])

model.eval()
with torch.no_grad():
    test_loss = 0
    n_batch = 0
    n_correct = 0
    n_samples = 0

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    for input_ids, attention_mask, target in tqdm(test_loader, desc="Test", unit="batch"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = output_layer(preds)

        loss = criterion(preds, target)

        test_loss += loss.item()
        n_batch += 1

        result = torch.argmax(preds, dim=1) 
        n_samples += target.size(0)
        n_correct += (result == target).sum().item()

        for prediction, ground_truth in zip(result, target):
            if prediction == ground_truth:
                each_label_correct[ground_truth.item()] += 1
            each_label_total[ground_truth.item()] += 1

    test_loss /= n_batch
    acc = 100.0 * n_correct / n_samples
    logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [test_loss], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.2f}%')

    for label, total_count in each_label_total.items():
        correct_count = each_label_correct.get(label, 0)  
        false_count = total_count - correct_count
        classification_report = pd.concat([classification_report, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
        print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

if not os.path.exists('logs'):
    os.makedirs('logs')

logger.to_csv('logs/flat_metrics.csv', index=False, encoding='utf-8')
classification_report.to_csv('logs/flat_classification_report.csv', index=False, encoding='utf-8')


# create graph
logger = pd.read_csv("logs/flat_metrics.csv", dtype={'accuracy': float, 'loss': float})

train_log = logger[logger['stage'] == 'train']
valid_log = logger[logger['stage'] == 'valid']

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.plot(train_log['epoch'], train_log['accuracy'], marker='o', label='Train Accuracy')
plt.plot(valid_log['epoch'], valid_log['accuracy'], marker='o', label='Validation Accuracy')

best_train_accuracy = train_log['accuracy'].max()
best_valid_accuracy = valid_log['accuracy'].max()

plt.annotate('best', xy=(train_log['epoch'][train_log['accuracy'].idxmax()], best_train_accuracy), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
plt.annotate('best', xy=(valid_log['epoch'][valid_log['accuracy'].idxmax()], best_valid_accuracy), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

plt.title(f'Best Training Accuracy: {best_train_accuracy:.2f} | Best Validation Accuracy: {best_valid_accuracy:.2f}', ha='center', fontsize='medium')
plt.legend()
plt.savefig('logs/flat_accuracy_metrics.png')
plt.clf()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.plot(train_log['epoch'], train_log['loss'], marker='o', label='Train Loss')
plt.plot(valid_log['epoch'], valid_log['loss'], marker='o', label='Validation Loss')

best_train_loss = train_log['loss'].min()
best_valid_loss = valid_log['loss'].min()

plt.annotate('best', xy=(train_log['epoch'][train_log['loss'].idxmin()], best_train_loss), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
plt.annotate('best', xy=(valid_log['epoch'][valid_log['loss'].idxmin()], best_valid_loss), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

plt.title(f'Best Training Loss: {best_train_loss:.2f} | Best Validation Loss: {best_valid_loss:.2f}', ha='center', fontsize='medium')
plt.legend()
plt.savefig('logs/flat_loss_metrics.png')
plt.clf()