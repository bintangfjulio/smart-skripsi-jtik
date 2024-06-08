import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from util.preprocessor import Preprocessor
from util.hyperparameter import get_hyperparameters
from sklearn.metrics.pairwise import cosine_similarity


dataset = pd.read_json("dataset/init_data_repo_jtik.json")
dataset = dataset[['nama_pembimbing', 'url', 'judul', 'abstrak', 'kata_kunci']]

config = get_hyperparameters()
preprocesor = Preprocessor(bert_model=config["bert_model"], max_length=config["max_length"])

tqdm.pandas(desc="Preprocessing Stage")
dataset["preprocessed"] = dataset.progress_apply(lambda data: preprocesor.text_processing(data), axis=1)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset["preprocessed"])
attribut = dataset[['judul', 'abstrak', 'kata_kunci', 'nama_pembimbing', 'url']].to_dict(orient='records')

with open('dataset/tfidf_data.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'attribut': attribut
    }, f)


# with open('dataset/tfidf_data.pkl', 'rb') as f:
#     tfidf_data = pickle.load(f)

# vectorizer = tfidf_data['vectorizer']
# tfidf_matrix = tfidf_data['tfidf_matrix']
# attribut = tfidf_data['attribut']

# def content_based_filtering(data):
#     preprocessed = preprocesor.text_processing(data)
#     matrix = vectorizer.transform([preprocessed])

#     similarity_scores = cosine_similarity(matrix, tfidf_matrix).flatten()

#     score_indices = similarity_scores.argsort()[::-1]
#     top_indices = score_indices[:3]
#     top_similarity = [(index, similarity_scores[index]) for index in top_indices]

#     attribut_recommended = [attribut[idx] for idx, _ in top_similarity]

#     return attribut_recommended, top_similarity


# attribut_recommended, top_similarity = content_based_filtering({
#     "abstrak": "",
#     "kata_kunci": ""
# })

# for idx, (attribut, score) in enumerate(zip(attribut_recommended, top_similarity)):
#     print(f"\nRank {idx + 1}:")
#     print(f"Similarity Score: {score[1]}")
#     print(f"Title: {attribut['judul']}")
#     print(f"Abstract: {attribut['abstrak']}")
#     print(f"Keywords: {attribut['kata_kunci']}")
#     print(f"Supervisor: {attribut['nama_pembimbing']}")
#     print(f"URL: {attribut['url']}")