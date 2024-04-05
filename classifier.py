# import
import argparse
import emoji
import re
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="abstrak")
parser.add_argument("--dataset", type=str, default='data_skripsi_jtik.csv')
parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--max_length", type=int, default=360)
config = vars(parser.parse_args())

text = input('Insert text to classify: ')

dataset = pd.read_csv(f'datasets/{config["dataset"]}')
stop_words = StopWordRemoverFactory().get_stop_words()
tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
stemmer = StemmerFactory().create_stemmer()
labels = sorted(dataset['prodi'].unique().tolist())


# preprocessor
text = text.lower()
text = emoji.replace_emoji(text, replace='') 
text = re.sub(r'\n', ' ', text) 
text = re.sub(r'http\S+', '', text)  
text = re.sub(r'\d+', '', text)  
text = re.sub(r'[^a-zA-Z ]', '', text)  
text = ' '.join([word for word in text.split() if word not in stop_words])  
text = stemmer.stem(text)
text = text.strip()   

token = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=config["max_length"],
        return_tensors='pt',
        padding="max_length", 
        truncation=True)


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
        bert_hidden_states = bert_output.hidden_states
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
pretrained_model = torch.load('checkpoints/tes.pt', map_location=device)

# print("Loading Checkpoint from Epoch", pretrained_model['epoch'])
print(pretrained_model['model_state'].keys())
model.load_state_dict(pretrained_model['model_state'])
model.to(device)


# classification
model.eval()
with torch.no_grad():
    preds = model(input_ids=token["input_ids"].to(device), attention_mask=token["attention_mask"].to(device))
    result = torch.argmax(preds, dim=1)
    print(labels[result[0]]) 