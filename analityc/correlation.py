import pandas as pd
from scipy.stats import pearsonr


uat = pd.read_csv("data_uat.csv")
sus = pd.read_csv("data_sus.csv")
nps = pd.read_csv("data_nps.csv")

x = uat['APW']
# y = sus['SUS']
y = nps['NPS']

correlation, _ = pearsonr(x, y)

print(f"Koefisien Korelasi Pearson: {correlation * 100}")