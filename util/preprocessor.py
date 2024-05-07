import emoji
import re
import torch

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from torch.utils.data import TensorDataset


class Preprocessor:
    def __init__(self, bert_model, max_length):
        self.stop_words = StopWordRemoverFactory().get_stop_words()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, use_fast=False)
        self.stemmer = StemmerFactory().create_stemmer()
        self.max_length = max_length

    def text_processing(self, row):
        text = str(row["kata_kunci"]) + " - " + str(row["abstrak"])
        text = text.lower()
        text = emoji.replace_emoji(text, replace='') 
        text = re.sub(r'\n', ' ', text) 
        text = re.sub(r'http\S+', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^a-zA-Z ]', '', text)  
        text = ' '.join([word for word in text.split() if word not in self.stop_words])  
        text = self.stemmer.stem(text)
        text = text.strip()      
        token = self.tokenizer(text=text, max_length=self.max_length, padding="max_length", truncation=True) 

        return token['input_ids'], token['attention_mask']
    
    def get_labels(self, dataset):
        labels = dataset["target"].unique().tolist()
        labels = sorted(dataset)
        
        return labels
    
    def train_test_split(self, dataset, train_percentage):
        input_ids = torch.tensor(dataset['input_ids'])
        attention_mask = torch.tensor(dataset['attention_mask'])
        target = torch.tensor(dataset['target'])

        tensor_dataset = TensorDataset(input_ids, attention_mask, target)

        train_size = round(len(tensor_dataset) * train_percentage)
        test_size = len(tensor_dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(tensor_dataset, [train_size, test_size])

        return train_set, test_set

    def train_valid_split(self, train_set, train_percentage):
        train_size = round(len(train_set) * train_percentage)
        valid_size = len(train_set) - train_size

        train_set, valid_set = torch.utils.data.random_split(train_set, [train_size, valid_size])

        return train_set, valid_set

