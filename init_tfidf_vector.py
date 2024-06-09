import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from util.preprocessor import Preprocessor
from util.hyperparameter import get_hyperparameters


dataset = pd.read_json("dataset/init_data_repo_jtik.json")
dataset = dataset[['nama_pembimbing', 'url', 'judul', 'abstrak', 'kata_kunci']]

config = get_hyperparameters()
preprocessor = Preprocessor(bert_model=config["bert_model"], max_length=config["max_length"])

tqdm.pandas(desc="Preprocessing Stage")
dataset["preprocessed"] = dataset.progress_apply(lambda data: preprocessor.text_processing(data), axis=1)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset["preprocessed"])
attribut = dataset[['judul', 'abstrak', 'kata_kunci', 'nama_pembimbing', 'url']].to_dict(orient='records')

with open('dataset/tfidf_data.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'attribut': attribut
    }, f)