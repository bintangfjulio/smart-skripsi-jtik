import pandas as pd

df = pd.read_csv("dataset/data_repo_jtik.csv")
nama_pembimbing_counts = df["nama_pembimbing"].value_counts().sort_index()
print(nama_pembimbing_counts)
