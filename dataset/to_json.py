import pandas as pd

df = pd.read_csv("data_repo_jtik.csv")
df = df[df['nama_pembimbing'] != '-']
df['kompetensi'] = ""

counts = df["prodi"].value_counts().sort_index()
print(len(df))
print(counts)

df.to_json("init_data_repo_jtik.json", orient="records")
