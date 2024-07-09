import pandas as pd


stage = "test"


model = "IndoBERT"
# model = "IndoBERTweet"
# model = "IndoRoBERTa"
lr = "4e-05"
batch_size = "16"

df = pd.read_csv(f"{model}_{lr}_{batch_size}_0.1/metrics.csv")
df = df[df['stage'] == stage]
df = df[["accuracy", "precision", "recall", "f1"]] 

print(df)
