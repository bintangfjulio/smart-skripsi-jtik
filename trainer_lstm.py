# import
import random
import os
import torch
import numpy as np
import multiprocessing
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from model.lstm import LSTM


# setup
parser = argparse.ArgumentParser()
parser.add_argument("--bidirectional", type=bool, default=False, help='Bi-LSTM T/F')
config = vars(parser.parse_args())

if(config['bidirectional']):
    folder_path = 'bilstm'

else:
    folder_path = 'lstm'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.display.float_format = '{:,.2f}'.format  

np.random.seed(42) 
torch.manual_seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

dataset = pd.read_pickle("dataset/preprocessed_set.pkl")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset['preprocessed']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['kelompok_bidang_keahlian'])

input_ids = torch.tensor(X, dtype=torch.float32)
target = torch.tensor(y, dtype=torch.long)
tensor_dataset = TensorDataset(input_ids, target)

train_valid_size = round(len(tensor_dataset) * (1.0 - 0.2))
test_size = len(tensor_dataset) - train_valid_size
train_valid_set, test_set = torch.utils.data.random_split(tensor_dataset, [train_valid_size, test_size])

train_size = round(len(train_valid_set) * (1.0 - 0.1))
valid_size = len(train_valid_set) - train_size
train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_size, valid_size])


train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                        batch_size=16, 
                                        shuffle=True,
                                        num_workers=multiprocessing.cpu_count())

valid_loader = torch.utils.data.DataLoader(dataset=valid_set, 
                                        batch_size=16, 
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                        batch_size=16, 
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())


# fine-tune
model = LSTM(input_size=input_ids.shape[1], bidirectional=config['bidirectional'], num_classes=len(label_encoder.classes_))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

best_loss = 9.99
failed_counter = 0

graph_logger = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'loss', 'epoch', 'stage']) 

optimizer.zero_grad()
model.zero_grad()

for epoch in range(2000):
    if failed_counter == 3:
        print("Early Stopping")
        break

    train_loss = 0
    n_batch = 0

    y_true_train = []
    y_pred_train = []

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    model.train(True)
    for input_ids, target in tqdm(train_loader, desc="Training Stage", unit="batch"):
        input_ids = input_ids.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids)
        loss = criterion(preds, target)

        train_loss += loss.item()
        n_batch += 1

        result = torch.argmax(preds, dim=1) 

        y_pred_train.extend(result.cpu().numpy())
        y_true_train.extend(target.cpu().numpy())

        for prediction, ground_truth in zip(result, target):
            if prediction == ground_truth:
                each_label_correct[ground_truth.item()] += 1
            each_label_total[ground_truth.item()] += 1

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        model.zero_grad()

    train_loss /= n_batch

    y_true_train = np.array(y_true_train)
    y_pred_train = np.array(y_pred_train)

    accuracy = accuracy_score(y_true_train, y_pred_train)
    precision = precision_score(y_true_train, y_pred_train, average='weighted')
    recall = recall_score(y_true_train, y_pred_train, average='weighted')
    f1 = f1_score(y_true_train, y_pred_train, average='weighted')

    graph_logger = pd.concat([graph_logger, pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'loss': [train_loss], 'epoch': [epoch+1], 'stage': ['train']})], ignore_index=True)
    print(f'Epoch [{epoch + 1}/{2000}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}%, Training Precision: {precision:.2f}%, Training Recall: {recall:.2f}%, Training F1: {f1:.2f}%')

    model.eval()
    with torch.no_grad():
        val_loss = 0
        n_batch = 0

        y_true_valid = []
        y_pred_valid = []

        each_label_correct = defaultdict(int)
        each_label_total = defaultdict(int)

        for input_ids, target in tqdm(valid_loader, desc="Validation Stage", unit="batch"):
            input_ids = input_ids.to(device)
            target = target.to(device)

            preds = model(input_ids=input_ids)
            loss = criterion(preds, target)

            val_loss += loss.item()
            n_batch += 1

            result = torch.argmax(preds, dim=1) 

            y_pred_valid.extend(result.cpu().numpy())
            y_true_valid.extend(target.cpu().numpy())

            for prediction, ground_truth in zip(result, target):
                if prediction == ground_truth:
                    each_label_correct[ground_truth.item()] += 1
                each_label_total[ground_truth.item()] += 1

            optimizer.zero_grad()
            model.zero_grad()

        val_loss /= n_batch

        y_true_valid = np.array(y_true_valid)
        y_pred_valid = np.array(y_pred_valid)

        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        precision = precision_score(y_true_valid, y_pred_valid, average='weighted')
        recall = recall_score(y_true_valid, y_pred_valid, average='weighted')
        f1 = f1_score(y_true_valid, y_pred_valid, average='weighted')

        graph_logger = pd.concat([graph_logger, pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'loss': [val_loss], 'epoch': [epoch+1], 'stage': ['valid']})], ignore_index=True)
        print(f'Epoch [{epoch + 1}/{2000}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation Precision: {precision:.2f}%, Validation Recall: {recall:.2f}%, Validation F1: {f1:.2f}%')
        
        if round(val_loss, 2) < round(best_loss, 2):
            if not os.path.exists(f'checkpoint/{folder_path}'):
                os.makedirs(f'checkpoint/{folder_path}')

            if os.path.exists(f'checkpoint/{folder_path}/pretrained_classifier.pt'):
                os.remove(f'checkpoint/{folder_path}/pretrained_classifier.pt')

            torch.save(model.state_dict(), f'checkpoint/{folder_path}/pretrained_classifier.pt')

            best_loss = val_loss
            failed_counter = 0

        else:
            failed_counter += 1

checkpoint = torch.load(f'checkpoint/{folder_path}/pretrained_classifier.pt', map_location=device)
model.load_state_dict(checkpoint)

model.eval()
with torch.no_grad():
    test_loss = 0
    n_batch = 0

    y_true_test = []
    y_pred_test = []

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    for input_ids, target in tqdm(test_loader, desc="Test Stage", unit="batch"):
        input_ids = input_ids.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids)
        loss = criterion(preds, target)

        test_loss += loss.item()
        n_batch += 1

        result = torch.argmax(preds, dim=1) 

        y_pred_test.extend(result.cpu().numpy())
        y_true_test.extend(target.cpu().numpy())

        for prediction, ground_truth in zip(result, target):
            if prediction == ground_truth:
                each_label_correct[ground_truth.item()] += 1
            each_label_total[ground_truth.item()] += 1

    test_loss /= n_batch

    y_true_test = np.array(y_true_test)
    y_pred_test = np.array(y_pred_test)

    accuracy = accuracy_score(y_true_test, y_pred_test)
    precision = precision_score(y_true_test, y_pred_test, average='weighted')
    recall = recall_score(y_true_test, y_pred_test, average='weighted')
    f1 = f1_score(y_true_test, y_pred_test, average='weighted')

    graph_logger = pd.concat([graph_logger, pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'loss': [test_loss], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test Precision: {precision:.2f}%, Test Recall: {recall:.2f}%, Test F1: {f1:.2f}%')

if not os.path.exists('log'):
    os.makedirs('log')

graph_logger.to_csv(f'log/{folder_path}/metrics.csv', index=False, encoding='utf-8')

# export result
graph_logger = pd.read_csv(f"log/{folder_path}/metrics.csv", dtype={'accuracy': float, 'precision': float, 'recall': float, 'f1': float, 'loss': float})

train_log = graph_logger[graph_logger['stage'] == 'train']
valid_log = graph_logger[graph_logger['stage'] == 'valid']

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.plot(train_log['epoch'], train_log[metric], marker='o', label=f'Train {metric.capitalize()}')
    plt.plot(valid_log['epoch'], valid_log[metric], marker='o', label=f'Validation {metric.capitalize()}')

    best_train = train_log[metric].max()
    best_valid = valid_log[metric].max()

    plt.annotate('best', xy=(train_log['epoch'][train_log[metric].idxmax()], best_train), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    plt.annotate('best', xy=(valid_log['epoch'][valid_log[metric].idxmax()], best_valid), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

    plt.title(f'Best Training {metric.capitalize()}: {best_train:.2f} | Best Validation {metric.capitalize()}: {best_valid:.2f}', ha='center', fontsize='medium')
    plt.legend()
    plt.savefig(f'log/{folder_path}/{metric}_metrics.png')
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
plt.savefig(f'log/{folder_path}/loss_metrics.png')
plt.clf()