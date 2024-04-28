import argparse
import emoji
import re
import torch
import pandas as pd

from torch import clamp
from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from model.bert_cnn import BERT_CNN


def preprocessor(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='') 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^a-zA-Z ]', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    text = stemmer.stem(text)
    text = text.strip()   
    
    return text

def classification(text):
    token = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=config["max_length"],
        return_tensors='pt',
        padding="max_length", 
        truncation=True)
    
    pretrained_bert = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=True)
    model = model = BERT_CNN(num_classes=len(labels), pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
    model.load_state_dict(torch.load('checkpoints/model_result.pt', map_location=device))
    model.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(input_ids=token["input_ids"].to(device), attention_mask=token["attention_mask"].to(device))
        result = torch.argmax(preds, dim=1)

    return labels[result[0]]

def similarity_checker(text_1, text_2):
    token = tokenizer([text_1, text_2], max_length=config['max_length'], padding="max_length", truncation=True,  return_tensors='pt') 
    model = BertModel.from_pretrained(config["bert_model"], output_attentions=False, output_hidden_states=False)

    attention_mask = token.attention_mask

    outputs = model(**token)
    embeddings = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask

    summed = masked_embeddings.sum(1)
    counts = clamp(mask.sum(1), min=1e-9)
    mean_pooled = (summed / counts).detach().numpy()

    similary = cosine_similarity([mean_pooled[0]], [mean_pooled[1]])   
    
    return similary[0][0]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='data_repo_jtik.csv')
    parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=360)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_bert_states", type=int, default=4)
    config = vars(parser.parse_args())

    dataset = pd.read_csv(f'dataset/{config["dataset"]}')
    stop_words = StopWordRemoverFactory().get_stop_words()
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"], use_fast=False)
    stemmer = StemmerFactory().create_stemmer()
    labels = sorted(dataset['prodi'].unique().tolist())

    responses = [{}]
    
    abstract = input("Please input abstract: ")
    abstract = preprocessor(abstract)
    
    print("Classification...")
    classified = classification(abstract)
    print(f"\nClassification Result: {classified}")

    print("Similarity Check...")
    highest_dosen_similarity = {}
    for item in responses:
        abstrak_response = preprocessor(item["abstrak"])
        similiraty_score = similarity_checker(abstract, abstrak_response)

        if item["dosen"] in highest_dosen_similarity:
            if similiraty_score > highest_dosen_similarity[item["dosen"]]:
                highest_dosen_similarity[item["dosen"]] = similiraty_score
        else:
            highest_dosen_similarity[item["dosen"]] = similiraty_score

    sorted_scores = sorted(highest_dosen_similarity.items(), key=lambda x: x[1], reverse=True)
    for i, (dosen, score) in enumerate(sorted_scores):
        print(f"{i+1}. Dosen: {dosen}, Similarity Score: {score}")