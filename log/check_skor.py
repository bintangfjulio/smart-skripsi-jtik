import pandas as pd


stage = "test"


# model = "IndoBERT"
# model = "IndoBERTweet"
model = "IndoRoBERTa"
batch_size = "16"
lr = "3e-05"

df = pd.read_csv(f"{model}_{lr}_{batch_size}_0.1/metrics.csv")
df = df[df['stage'] == stage]
df = df[["accuracy", "precision", "recall", "f1"]] 

print(df)
