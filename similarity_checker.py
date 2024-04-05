# import
import argparse
import re
import emoji

from torch import clamp
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# setup
parser = argparse.ArgumentParser()
parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased")
parser.add_argument("--max_length", type=int, default=375)
config = vars(parser.parse_args())

tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
stop_words = StopWordRemoverFactory().get_stop_words()
stemmer = StemmerFactory().create_stemmer()
model = BertModel.from_pretrained(config["bert_model"])

sentences = [input("Please enter first text: "), input("Please enter second text: ")]


# preprocessor
for index, text in enumerate(sentences):
    text = str(text) 
    text = text.lower()
    text = emoji.replace_emoji(text, replace='') 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^a-zA-Z ]', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    text = stemmer.stem(text)
    text = text.strip()  
    sentences[index] = text


# feature extraction
token = tokenizer([sentences[0], sentences[1]], max_length=config['max_length'], padding="max_length", truncation=True,  return_tensors='pt') 
attention_mask = token.attention_mask

outputs = model(**token)
embeddings = outputs.last_hidden_state

mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
masked_embeddings = embeddings * mask

summed = masked_embeddings.sum(1)
counts = clamp(mask.sum(1), min=1e-9)
mean_pooled = (summed / counts).detach().numpy()

similary = cosine_similarity([mean_pooled[0]], [mean_pooled[1]])    
print(similary[0][0])