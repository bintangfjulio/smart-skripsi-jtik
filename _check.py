import pandas as pd

df = pd.read_csv("dataset/data_repo_jtik.csv")
# df = df[df['nama_pembimbing'] != '-']
counts = df["prodi"].value_counts().sort_index()
print(len(df))
print(counts)
