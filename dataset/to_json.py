import pandas as pd

df = pd.read_csv("data_repo_jtik.csv")
df = df[df['nama_pembimbing'] != '-']
df['kompetensi'] = ""

def generate_id_sequence_prodi(group):
    group['id_sequence_prodi'] = range(1, len(group) + 1)
    return group

df = df.groupby('prodi').apply(generate_id_sequence_prodi)

print(len(df))
print(df["prodi"].value_counts().sort_index())

# Save the dataframe to JSON
df.to_json("init_data_repo_jtik.json", orient="records")
