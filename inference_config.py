import torch
import emoji
import re
import pickle
import torch.nn as nn
import torch.nn.functional as F

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class BERT_CNN(nn.Module):
    def __init__(self, labels, pretrained_bert, window_sizes=[1, 2, 3, 4, 5], in_channels=4, out_channels=32, dropout=0.1, num_bert_states=4):
        super(BERT_CNN, self).__init__()
        self.pretrained_bert = AutoModel.from_pretrained(pretrained_bert, output_attentions=False, output_hidden_states=True)
        
        conv_layers = []
        for window_size in window_sizes:
            conv_layer = nn.Conv2d(in_channels, out_channels, (window_size, self.pretrained_bert.embeddings.word_embeddings.weight.size(1)))
            conv_layers.append(conv_layer)
            
        self.cnn = nn.ModuleList(conv_layers)

        self.dropout = nn.Dropout(dropout) 
        self.num_bert_states = num_bert_states

        self.output_layer = nn.Linear(len(window_sizes) * out_channels, len(labels))

    def forward(self, input_ids, attention_mask):
        bert_output = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask)
        stacked_hidden_states = torch.stack(bert_output.hidden_states[-self.num_bert_states:], dim=1)

        pooling = []
        for layer in self.cnn:
            hidden_states = layer(stacked_hidden_states)
            relu_output = F.relu(hidden_states.squeeze(3))
            pooling.append(relu_output)

        max_pooling = []
        for features in pooling:
            pooled_features = F.max_pool1d(features, features.size(2)).squeeze(2)
            max_pooling.append(pooled_features)
        
        concatenated = torch.cat(max_pooling, dim=1)
        preds = self.dropout(concatenated)

        preds = self.output_layer(preds)
        
        return preds
    

class Inference():
    def __init__(self, max_length=360, pretrained_bert="indolem/indobert-base-uncased"):    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['Jaringan & IoT', 'Multimedia & Teknologi: AI Game', 'Rekayasa Perangkat Lunak', 'Sistem Cerdas']

        self.stop_words = StopWordRemoverFactory().get_stop_words()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert, use_fast=False)
        self.stemmer = StemmerFactory().create_stemmer()
        self.max_length = max_length
        
        self.model = BERT_CNN(labels=self.labels, pretrained_bert=pretrained_bert)
        checkpoint = torch.load("checkpoint/pretrained_classifier.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        with open('checkpoint/pretrained_tfidf.pkl', 'rb') as f:
            tfidf_data = pickle.load(f)

        self.vectorizer = tfidf_data['vectorizer']
        self.tfidf_matrix = tfidf_data['tfidf_matrix']
        self.attribut = tfidf_data['attribut']

    def text_processing(self, abstrak, kata_kunci):
        text = str(kata_kunci) + " - " + str(abstrak)
        text = text.lower()
        text = emoji.replace_emoji(text, replace='') 
        text = re.sub(r'\n', ' ', text) 
        text = re.sub(r'http\S+', '', text)  
        text = re.sub(r'\d+', '', text)  
        text = re.sub(r'[^a-zA-Z ]', '', text)  
        text = ' '.join([word for word in text.split() if word not in self.stop_words])  
        text = self.stemmer.stem(text)
        text = text.strip()      

        return text

    def bert_tokenizer(self, text):
        token = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding="max_length", 
            truncation=True
        )

        return token['input_ids'], token['attention_mask']

    def classification(self, data):
        input_ids, attention_mask = self.bert_tokenizer(data)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            result = torch.softmax(preds, dim=1)[0]

            probs = {}
            for index, prob in enumerate(result):
                probs[self.labels[index]] = round(prob.item() * 100, 2)

            highest_prob = torch.argmax(preds, dim=1)

            kbk = self.labels[highest_prob]

        return probs, kbk
    
    def content_based_filtering(self, data):
        matrix = self.vectorizer.transform([data])

        similarity_scores = cosine_similarity(matrix, self.tfidf_matrix).flatten()

        score_indices = similarity_scores.argsort()[::-1]
        top_indices = score_indices[:3]
        top_similarity = [(index, similarity_scores[index]) for index in top_indices]

        attribut_recommended = [self.attribut[idx] for idx, _ in top_similarity]

        recommended = []
        for idx, (attribut, score) in enumerate(zip(attribut_recommended, top_similarity)):
            result = {
                "rank": idx + 1,
                "similarity_score": round(score[1] * 100, 2),
                "title": attribut['judul'],
                "abstract": attribut['abstrak'],
                "keywords": attribut['kata_kunci'],
                "supervisor": attribut['nama_pembimbing'],
                "url": attribut['url']
            }

            recommended.append(result)

        return recommended