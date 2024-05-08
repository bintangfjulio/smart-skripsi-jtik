# import
import argparse
import random
import os
import torch
import numpy as np
import multiprocessing
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from torch.utils.data import TensorDataset
from transformers import BertModel
from tqdm import tqdm
from collections import defaultdict
from model.bert_cnn import BERT_CNN
from util.preprocessor import Preprocessor


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.options.display.float_format = '{:,.2f}'.format  

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='prodi', help='Target Root Column')
parser.add_argument("--node", type=str, default='nama_pembimbing', help='Target Node Column')
parser.add_argument("--dataset", type=str, default='init_data_repo_jtik.json', help='Dataset Path')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size')
parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased", help='BERT Model')
parser.add_argument("--seed", type=int, default=42, help='Random Seed')
parser.add_argument("--max_epochs", type=int, default=30, help='Number of Epochs')
parser.add_argument("--lr", type=float, default=2e-5, help='Learning Rate')
parser.add_argument("--dropout", type=float, default=0.1, help='Dropout')
parser.add_argument("--patience", type=int, default=3, help='Patience')
parser.add_argument("--num_bert_states", type=int, default=4, help='Number of BERT Last States')
parser.add_argument("--max_length", type=int, default=360, help='Max Length')
parser.add_argument("--in_channels", type=int, default=4, help='CNN In Channels')
parser.add_argument("--out_channels", type=int, default=32, help='CNN Out Channels')
parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5], help='CNN Kernel')

config = vars(parser.parse_args())

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
    dataset[['input_ids', 'attention_mask']] = dataset.progress_apply(lambda data: preprocessor.text_processing(data), axis=1, result_type='expand')
    dataset.to_pickle("dataset/preprocessed_set.pkl")

dataset = pd.read_pickle("dataset/preprocessed_set.pkl")
root_labels, node_labels = preprocessor.get_grouped_labels(dataset=dataset, root=config["root"], node=config["node"])

dataset["root"] = dataset[config["root"]].apply(lambda data: root_labels[data])
dataset["node"] = ""

for key in root_labels.keys():
    grouped_dataset = dataset[dataset[config["root"]] == key]
    grouped_dataset.loc[:, "node"] = grouped_dataset[config["node"]].apply(lambda data: node_labels[key].index(data))
    dataset.update(grouped_dataset)

dataset = dataset.sample(frac=1)
dataset_size = dataset.shape[0]
train_valid_size = round(dataset_size * 0.8)

train_valid_set = pd.DataFrame(dataset.iloc[:train_valid_size, :])
test_set = pd.DataFrame(dataset.iloc[train_valid_size:, :])

input_ids = torch.tensor(test_set['input_ids'].tolist())
attention_mask = torch.tensor(test_set['attention_mask'].tolist())
root = torch.tensor(test_set['root'].tolist())
node = torch.tensor(test_set["node"].tolist())
test_set = TensorDataset(input_ids, attention_mask, root, node)

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                        batch_size=config["batch_size"], 
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())

def fit(section, train_loader, valid_loader):
    if section == "root":
        labels = root_labels

    else:
        labels = node_labels[section]
    
    model = BERT_CNN(pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
    model.to(device)

    output_layer = nn.Linear(len(config["window_sizes"]) * config["out_channels"], len(labels))
    output_layer.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_loss = 9.99
    failed_counter = 0

    logger = pd.DataFrame(columns=['accuracy', 'loss', 'epoch', 'stage', 'section']) 
    classification_report = pd.DataFrame(columns=['label', 'correct_prediction', 'false_prediction', 'total_prediction', 'epoch', 'stage', 'section'])

    if section == "root":
        checkpoint = {}

    else:
        checkpoint = torch.load('checkpoint/hierarchy_model.pt', map_location=device)
        model.load_state_dict(checkpoint["root_hidden_states"])

    optimizer.zero_grad()
    model.zero_grad()
    output_layer.zero_grad()

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
        for input_ids, attention_mask, root, node in tqdm(train_loader, desc="Training Stage", unit="batch"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if section == "root":
                target = root.to(device)

            else:
                target = node.to(device)

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
        logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [train_loss], 'epoch': [epoch+1], 'stage': ['train'], 'section': [section]})], ignore_index=True)
        print(f'Section: {section}, Epoch [{epoch + 1}/{config["max_epochs"]}], Training Loss: {train_loss:.4f}, Training Accuracy: {acc:.2f}%')

        for label, total_count in each_label_total.items():
            correct_count = each_label_correct.get(label, 0)  
            false_count = total_count - correct_count

            if(section == "root"):
                y = root_labels[label]

            else: 
                y = node_labels[section].index(label)

            classification_report = pd.concat([classification_report, pd.DataFrame({'label': [y], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['train'], 'section': [section]})], ignore_index=True)
            print(f"Section: {section}, Label: {y}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

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
                
                if section == "root":
                    target = root.to(device)

                else:
                    target = node.to(device)

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
            logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [val_loss], 'epoch': [epoch+1], 'stage': ['valid'], 'section': [section]})], ignore_index=True)
            print(f'Section: {section}, Epoch [{epoch + 1}/{config["max_epochs"]}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc:.2f}%')

            for label, total_count in each_label_total.items():
                correct_count = each_label_correct.get(label, 0)  
                false_count = total_count - correct_count

                if(section == "root"):
                    y = root_labels[label]

                else: 
                    y = node_labels[section].index(label)

                classification_report = pd.concat([classification_report, pd.DataFrame({'label': [y], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [epoch+1], 'stage': ['valid'], 'section': [section]})], ignore_index=True)
                print(f"Section: {section}, Label: {y}, Correct Predictions: {correct_count}, False Predictions: {false_count}")
            
            if round(val_loss, 2) < round(best_loss, 2):
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')

                if os.path.exists('checkpoint/hierarchy_model.pt'):
                    os.remove('checkpoint/hierarchy_model.pt')

                checkpoint[f'{section.lower().replace(" ", "_")}_hidden_states'] = model.state_dict()
                checkpoint[f'{section.lower().replace(" ", "_")}_last_hidden_state'] = output_layer.state_dict()

                torch.save(checkpoint, 'checkpoint/hierarchy_model.pt')

                best_loss = val_loss
                failed_counter = 0

            else:
                failed_counter += 1

    # checkpoint = torch.load(f'checkpoint/hierarchy_model.pt', map_location=device)
    # model.load_state_dict(checkpoint["root_hidden_states"])
    # output_layer.load_state_dict(checkpoint["root_last_hidden_state"])

    # if not os.path.exists('log'):
    #     os.makedirs('log')

    # logger.to_csv(f'log/hierarchy_metrics.csv', index=False, encoding='utf-8')
    # classification_report.to_csv(f'log/hierarchy_classification_report.csv', index=False, encoding='utf-8')


    # convert graph
    # logger = pd.read_csv(f"log/flat_{config['target']}_metrics.csv", dtype={'accuracy': float, 'loss': float})

    # train_log = logger[logger['stage'] == 'train']
    # valid_log = logger[logger['stage'] == 'valid']

    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    # plt.plot(train_log['epoch'], train_log['accuracy'], marker='o', label='Train Accuracy')
    # plt.plot(valid_log['epoch'], valid_log['accuracy'], marker='o', label='Validation Accuracy')

    # best_train_accuracy = train_log['accuracy'].max()
    # best_valid_accuracy = valid_log['accuracy'].max()

    # plt.annotate('best', xy=(train_log['epoch'][train_log['accuracy'].idxmax()], best_train_accuracy), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    # plt.annotate('best', xy=(valid_log['epoch'][valid_log['accuracy'].idxmax()], best_valid_accuracy), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

    # plt.title(f'Best Training Accuracy: {best_train_accuracy:.2f} | Best Validation Accuracy: {best_valid_accuracy:.2f}', ha='center', fontsize='medium')
    # plt.legend()
    # plt.savefig(f'log/hierarchy_accuracy_metrics.png')
    # plt.clf()

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    # plt.plot(train_log['epoch'], train_log['loss'], marker='o', label='Train Loss')
    # plt.plot(valid_log['epoch'], valid_log['loss'], marker='o', label='Validation Loss')

    # best_train_loss = train_log['loss'].min()
    # best_valid_loss = valid_log['loss'].min()

    # plt.annotate('best', xy=(train_log['epoch'][train_log['loss'].idxmin()], best_train_loss), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
    # plt.annotate('best', xy=(valid_log['epoch'][valid_log['loss'].idxmin()], best_valid_loss), xytext=(-30, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

    # plt.title(f'Best Training Loss: {best_train_loss:.2f} | Best Validation Loss: {best_valid_loss:.2f}', ha='center', fontsize='medium')
    # plt.legend()
    # plt.savefig(f'log/hierarchy_loss_metrics.png')
    # plt.clf()

# def test(model, section, output_layer, criterion, test_loader, labels):
#     model.eval()
#     with torch.no_grad():
#         test_loss = 0
#         n_batch = 0
#         n_correct = 0
#         n_samples = 0

#         each_label_correct = defaultdict(int)
#         each_label_total = defaultdict(int)

#         for input_ids, attention_mask, target in tqdm(test_loader, desc="Test Stage", unit="batch"):
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)

#             if section == "root":
#                 target = root.to(device)

#             else:
#                 target = node.to(device)

#             preds = model(input_ids=input_ids, attention_mask=attention_mask)
#             preds = output_layer(preds)

#             loss = criterion(preds, target)

#             test_loss += loss.item()
#             n_batch += 1

#             result = torch.argmax(preds, dim=1) 
#             n_samples += target.size(0)
#             n_correct += (result == target).sum().item()

#             for prediction, ground_truth in zip(result, target):
#                 if prediction == ground_truth:
#                     each_label_correct[ground_truth.item()] += 1
#                 each_label_total[ground_truth.item()] += 1

#         test_loss /= n_batch
#         acc = 100.0 * n_correct / n_samples
#         logger = pd.concat([logger, pd.DataFrame({'accuracy': [acc], 'loss': [test_loss], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
#         print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.2f}%')

#         for label, total_count in each_label_total.items():
#             correct_count = each_label_correct.get(label, 0)  
#             false_count = total_count - correct_count

            # if(section == "root"):
            #     y = root_labels[label]

            # else: 
            #     y = node_labels[section].index(label)
#             classification_report = pd.concat([classification_report, pd.DataFrame({'label': [y], 'correct_prediction': [correct_count], 'false_prediction': [false_count], 'total_prediction': [total_count], 'epoch': [0], 'stage': ['test']})], ignore_index=True)
#             print(f"Label: {y}, Correct Predictions: {correct_count}, False Predictions: {false_count}")

def section_dataloader(dataset, section):
    if section != "root":
        dataset = dataset[dataset[config["root"]] == section]

    input_ids = torch.tensor(dataset['input_ids'].tolist())
    attention_mask = torch.tensor(dataset['attention_mask'].tolist())
    root = torch.tensor(dataset['root'].tolist())
    node = torch.tensor(dataset["node"].tolist())
    dataset = TensorDataset(input_ids, attention_mask, root, node)

    train_size = round(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                            batch_size=config["batch_size"], 
                                            shuffle=True,
                                            num_workers=multiprocessing.cpu_count())

    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, 
                                            batch_size=config["batch_size"], 
                                            shuffle=False,
                                            num_workers=multiprocessing.cpu_count())

    return train_loader, valid_loader

train_loader, valid_loader = section_dataloader(dataset=train_valid_set, section="root")
fit(section="root", train_loader=train_loader, valid_loader=valid_loader)

for key in root_labels.keys():
    train_loader, valid_loader = section_dataloader(dataset=train_valid_set, section=key)
    fit(section=key, train_loader=train_loader, valid_loader=valid_loader)