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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset
from transformers import BertModel
from tqdm import tqdm
from collections import defaultdict
from model.bert_cnn import BERT_CNN
from util.preprocessor import Preprocessor
from util.hyperparameter import get_hyperparameters
from sklearn.metrics import confusion_matrix


# setup
config = get_hyperparameters()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.display.float_format = '{:,.2f}'.format  

np.random.seed(config["seed"]) 
torch.manual_seed(config["seed"])
random.seed(config["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

dataset = pd.read_json(f'dataset/{config["dataset"]}')
pretrained_bert = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=True)
preprocessor = Preprocessor(bert_model=config["bert_model"], max_length=config["max_length"])


# preprocessor
if not os.path.exists("dataset/preprocessed_set.pkl"):
    tqdm.pandas(desc="Preprocessing Stage")
    dataset[['input_ids', 'attention_mask']] = dataset.progress_apply(lambda data: preprocessor.bert_tokenizer(preprocessor.text_processing(data)), axis=1, result_type='expand')
    dataset.to_pickle("dataset/preprocessed_set.pkl")

dataset = pd.read_pickle("dataset/preprocessed_set.pkl")
labels = preprocessor.get_labels(dataset=dataset, target=config["target"])
dataset["target"] = dataset[config["target"]].apply(lambda data: labels.index(data))

input_ids = torch.tensor(dataset['input_ids'].tolist())
attention_mask = torch.tensor(dataset['attention_mask'].tolist())
target = torch.tensor(dataset['target'].tolist())
tensor_dataset = TensorDataset(input_ids, attention_mask, target)

train_valid_size = round(len(tensor_dataset) * (1.0 - config["test_size"]))
test_size = len(tensor_dataset) - train_valid_size
train_valid_set, test_set = torch.utils.data.random_split(tensor_dataset, [train_valid_size, test_size])

train_size = round(len(train_valid_set) * (1.0 - config["valid_size"]))
valid_size = len(train_valid_set) - train_size
train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_size, valid_size])

# train_valid_set, test_set = preprocessor.train_test_split(dataset=dataset, test_size=config["test_size"])

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
model = BERT_CNN(labels=labels, pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

best_loss = 9.99
failed_counter = 0

graph_logger = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'loss', 'epoch', 'stage']) 
prediction_stats = pd.DataFrame(columns=['label', 'correct_prediction', 'false_prediction', 'total_prediction', 'epoch', 'stage'])

optimizer.zero_grad()
model.zero_grad()

for epoch in range(config["max_epochs"]):
    if failed_counter == config["patience"]:
        print("Early Stopping")
        break

    train_loss = 0
    n_batch = 0

    y_true_train = []
    y_pred_train = []

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    model.train(True)
    for input_ids, attention_mask, target in tqdm(train_loader, desc="Training Stage", unit="batch"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)

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
    print(f'Epoch [{epoch + 1}/{config["max_epochs"]}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}%, Training Precision: {precision:.2f}%, Training Recall: {recall:.2f}%, Training F1: {f1:.2f}%')

    for label, total_count in each_label_total.items():
        correct_count = each_label_correct.get(label, 0)  
        false_count = total_count - correct_count
        prediction_stats = pd.concat([prediction_stats, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['train']})], ignore_index=True)
        print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

    model.eval()
    with torch.no_grad():
        val_loss = 0
        n_batch = 0

        y_true_valid = []
        y_pred_valid = []

        each_label_correct = defaultdict(int)
        each_label_total = defaultdict(int)

        for input_ids, attention_mask, target in tqdm(valid_loader, desc="Validation Stage", unit="batch"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask)

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
        print(f'Epoch [{epoch + 1}/{config["max_epochs"]}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation Precision: {precision:.2f}%, Validation Recall: {recall:.2f}%, Validation F1: {f1:.2f}%')

        for label, total_count in each_label_total.items():
            correct_count = each_label_correct.get(label, 0)  
            false_count = total_count - correct_count
            prediction_stats = pd.concat([prediction_stats, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['valid']})], ignore_index=True)
            print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")
        
        if round(val_loss, 2) < round(best_loss, 2):
            if not os.path.exists('checkpoint'):
                os.makedirs('checkpoint')

            if os.path.exists(f'checkpoint/pretrained_classifier.pt'):
                os.remove(f'checkpoint/pretrained_classifier.pt')

            torch.save(model.state_dict(), f'checkpoint/pretrained_classifier.pt')

            best_loss = val_loss
            failed_counter = 0

        else:
            failed_counter += 1

checkpoint = torch.load(f'checkpoint/pretrained_classifier.pt', map_location=device)
model.load_state_dict(checkpoint)

model.eval()
with torch.no_grad():
    test_loss = 0
    n_batch = 0

    y_true_test = []
    y_pred_test = []

    each_label_correct = defaultdict(int)
    each_label_total = defaultdict(int)

    for input_ids, attention_mask, target in tqdm(test_loader, desc="Test Stage", unit="batch"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)

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

    for label, total_count in each_label_total.items():
        correct_count = each_label_correct.get(label, 0)  
        false_count = total_count - correct_count
        prediction_stats = pd.concat([prediction_stats, pd.DataFrame({'label': [labels[label]], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
        print(f"Label: {labels[label]}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.xticks(rotation=15, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    if not os.path.exists('log'):
        os.makedirs('log')

    plt.savefig('log/confusion_matrix.png', bbox_inches='tight')


graph_logger.to_csv(f'log/metrics.csv', index=False, encoding='utf-8')
prediction_stats.to_csv(f'log/prediction_stats.csv', index=False, encoding='utf-8')


# export result
graph_logger = pd.read_csv(f"log/metrics.csv", dtype={'accuracy': float, 'precision': float, 'recall': float, 'f1': float, 'loss': float})

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
    plt.savefig(f'log/{metric}_metrics.png')
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
plt.savefig(f'log/loss_metrics.png')
plt.clf()