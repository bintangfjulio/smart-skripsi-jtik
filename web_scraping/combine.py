import pandas as pd

df1 = pd.read_csv("jtik-newrepository/data_newrepo_jtik.csv")
df2 = pd.read_csv("jtik-oldrepository/data_oldrepo_jtik.csv")

df = pd.concat([df1, df2], ignore_index=True)

df['kata_kunci'] = df['kata_kunci'].str.lower()
df = df.drop_duplicates(subset="kata_kunci")

df['judul'] = df['judul'].str.lower()
df = df.drop_duplicates(subset="judul")

df['abstrak'] = df['abstrak'].str.lower()
df = df.drop_duplicates(subset="abstrak")

df.to_csv("data_skripsi_jtik.csv", index=False)