import argparse
import emoji
import re
import torch
import pandas as pd
import torch.nn as nn

from torch import clamp
from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from model.bert_cnn import BERT_CNN
from sentence_transformers import SentenceTransformer


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
    model = BERT_CNN(pretrained_bert=pretrained_bert, dropout=config["dropout"], window_sizes=config["window_sizes"], in_channels=config["in_channels"], out_channels=config["out_channels"], num_bert_states=config["num_bert_states"])
    output_layer = nn.Linear(len(config["window_sizes"]) * config["out_channels"], len(labels))

    checkpoint = torch.load('checkpoint/flat_prodi_model.pt', map_location=device)

    model.load_state_dict(checkpoint["hidden_states"])
    output_layer.load_state_dict(checkpoint["last_hidden_state"])

    model.to(device)
    output_layer.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(input_ids=token["input_ids"].to(device), attention_mask=token["attention_mask"].to(device))
        preds = output_layer(preds)
        result = torch.argmax(preds, dim=1)
        probs = torch.softmax(preds, dim=1)
        print(probs)

        stats = {}
        for index, prob in enumerate(probs[0]):
            stats[labels[index]] = round(prob.item() * 100, 1)

        print(stats)

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
    parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=360)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num_bert_states", type=int, default=4)
    config = vars(parser.parse_args())

    # dataset = pd.read_json(f'dataset/init_data_repo_jtik.json')
    stop_words = StopWordRemoverFactory().get_stop_words()
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"], use_fast=False)
    stemmer = StemmerFactory().create_stemmer()

    # labels = dataset["prodi"].unique().tolist()
    # labels = sorted(labels)
    
    # abstract = input("Please input abstract: ")
    abstract = "pemberian apresiasi terhadap kinerja karyawan dapat menjadi dorongan bagi karyawan lain agar termotivasi untuk meningkatkan kualitas kinerjanya. seperti yang dilakukan oleh pt united tractors tbk yang memberikan apresiasi terhadap internal facilitator terbaiknya. namun, penentuan internal facilitator terbaik ini masih dilakukan secara manual. oleh karena itu, dibuatlah sistem pendukung keputusan yang dapat membantu dalam mengambil keputusan internal facilitator terbaik dengan menggunakan metode simple multi attribute rating technique. metode smart dipilih karena berdasarkan fleksibilitas dan kesederhanaan yang dimiliki dalam memberikan keputusan. metode ini merupakan metode yang dapat menangani permasalahan dengan multi kriteria berdasarkan bobot kriteria dari setiap alternatif. berdasarkan hasil pengujian aplikasi, sistem ini dapat berjalan dengan sangat baik dan semestinya. sistem ini dapat memberikan hasil akhir pada perankingan internal facilitator terbaik dan membantu pihak terkait dalam memberikan keputusan dengan lebih efektif dan efisien."
    abstract = preprocessor(abstract)
    
    # print("Classification...")
    # classified = classification(abstract)
    # print(f"\nClassification Result: {classified}")

    
    responses = ["profil lulusan memiliki peran penting dalam menilai kompetensi sebuah perguruan tinggi, yang juga mempengaruhi akreditasi perguruan tinggi tersebut. tracer study adalah program yang dibuat oleh sekretariat direktorat jendral pendidikan tinggi pada tahun 2011 untuk memantau profil lulusan dari setiap perguruan tinggi di indonesia. tracer study bertujuan untuk melacak keberhasilan lulusan dalam mencapai kesuksesan di dunia kerja atau pengembangan karir setelah menyelesaikan pendidikan mereka. namun, di kampus politeknik negeri jakarta, terjadi tantangan dalam meningkatkan minat alumni untuk mengisi kuesioner tracer study dan dalam melacak alumni yang belum mengisi kuesioner. untuk mengatasi tantangan ini, pendekatan gamifikasi digunakan dengan menghadirkan elemen-elemen permainan dalam kuesioner untuk meningkatkan motivasi dan keterlibatan alumni dalam pengisian tracer study. fitur validasi data juga diperlukan untuk memastikan data yang diperoleh dari alumni adalah akurat dan dapat diandalkan. selain itu, dalam merancang aplikasi tracer study yang efektif, bahasa pemrograman php dan framework laravel dipilih untuk membangun aplikasi web yang dinamis dan interaktif."]

    print("Similarity Check...")
    highest_dosen_similarity = {}
    for item in responses:
        # abstrak_response = preprocessor(item)
        # similiraty_score = similarity_checker("aku suka kamu", "mesin mobil")
        # print(similiraty_score)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        sentences = [
            "aku suka kamu",
            "mesin mobil"
        ]

        embeddings = model.encode(sentences)
        print(embeddings.shape)

        similarities = model.similarity(embeddings, embeddings)
        print(similarities)

        # if item["dosen"] in highest_dosen_similarity:
        #     if similiraty_score > highest_dosen_similarity[item["dosen"]]:
        #         highest_dosen_similarity[item["dosen"]] = similiraty_score
        # else:
        #     highest_dosen_similarity[item["dosen"]] = similiraty_score

    # sorted_scores = sorted(highest_dosen_similarity.items(), key=lambda x: x[1], reverse=True)
    # for i, (dosen, score) in enumerate(sorted_scores):
    #     print(f"{i+1}. Dosen: {dosen}, Similarity Score: {score}")