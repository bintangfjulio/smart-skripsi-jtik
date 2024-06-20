import pickle
import os
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from util.preprocessor import Preprocessor
from util.hyperparameter import get_hyperparameters


config = get_hyperparameters()

dataset = pd.read_json(f"dataset/{config["dataset"]}")
dataset = dataset[['nama_pembimbing', 'url', 'judul', 'abstrak', 'kata_kunci']]

if not os.path.exists("dataset/preprocessed_set.pkl"):
    preprocessor = Preprocessor(bert_model=config["bert_model"], max_length=config["max_length"])
    tqdm.pandas(desc="Preprocessing Stage")
    dataset["preprocessed"] = dataset.progress_apply(lambda data: preprocessor.text_processing(data), axis=1)
    dataset.to_pickle("dataset/preprocessed_set.pkl")

dataset = pd.read_pickle("dataset/preprocessed_set.pkl")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset["preprocessed"])
attribut = dataset[['judul', 'abstrak', 'kata_kunci', 'nama_pembimbing', 'url']].to_dict(orient='records')

if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')

with open('checkpoint/pretrained_tfidf.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'attribut': attribut
    }, f)