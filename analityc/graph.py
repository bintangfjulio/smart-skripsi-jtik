import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# df = pd.read_csv("data_uat.csv")
# df = pd.read_csv("data_sus.csv")
df = pd.read_csv("data_nps.csv")

# by = "Program Studi"
by = "Jenis Kelamin"


# # KM
# avg_trust = df.groupby(by)['KM'].mean()
# plt.figure(figsize=(10, 6))
# bar_plot = avg_trust.plot(kind='bar', color='skyblue')
# plt.ylabel('Rata-rata')
# plt.title(f'Penilaian Kualitas Model Berdasarkan {by}')
# plt.xticks(rotation=0)  

# for i in bar_plot.containers:
#     bar_plot.bar_label(i, fmt='%.2f', label_type='edge')

# plt.tight_layout()
# plt.show()


# # KF
# means = df.groupby(by)[['KF1', 'KF2', 'KF3', 'KF4']].mean()
# soft_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

# fig, ax = plt.subplots(figsize=(12, 8))

# bar_width = 0.2
# bar_spacing = 0.05
# group_spacing = 0.5
# n_bars = len(means.columns)
# positions = np.arange(len(means)) * (n_bars * (bar_width + bar_spacing) + group_spacing)

# for i, col in enumerate(means.columns):
#     ax.bar(positions + i * (bar_width + bar_spacing), means[col], width=bar_width, label=col, color=soft_colors[i])

# ax.set_ylabel('Rata-rata')
# ax.set_title(f'Penilaian Kualitas Fitur Berdasarkan {by}')
# ax.set_xticks(positions + (n_bars * (bar_width + bar_spacing) - bar_spacing) / 2)
# ax.set_xticklabels(means.index, rotation=0)

# legend_labels = ['Authentifikasi', 'Klasifikasi', 'Riwayat', 'Ekspor Berkas']
# ax.legend(legend_labels, title='Fitur', bbox_to_anchor=(1.05, 1), loc='upper left')

# for container in ax.containers:
#     ax.bar_label(container, fmt='%.2f')

# plt.tight_layout()
# plt.show()


# # KW
# means = df.groupby(by)[['FW', 'AW', 'APW']].mean()
# soft_colors = ['#66c2a5', '#fc8d62', '#8da0cb']

# fig, ax = plt.subplots(figsize=(12, 8))

# bar_width = 0.2
# bar_spacing = 0.05
# group_spacing = 0.5
# n_bars = len(means.columns)
# positions = np.arange(len(means)) * (n_bars * (bar_width + bar_spacing) + group_spacing)

# for i, col in enumerate(means.columns):
#     ax.bar(positions + i * (bar_width + bar_spacing), means[col], width=bar_width, label=col, color=soft_colors[i])

# ax.set_ylabel('Rata-rata')
# ax.set_title(f'Penilaian Kualitas Web Berdasarkan {by}')
# ax.set_xticks(positions + (n_bars * (bar_width + bar_spacing) - bar_spacing) / 2)
# ax.set_xticklabels(means.index, rotation=0)

# legend_labels = ['Fungsional', 'Antarmuka', 'Alur Penggunaan']
# ax.legend(legend_labels, title='Kualitas', bbox_to_anchor=(1.05, 1), loc='upper left')

# for container in ax.containers:
#     ax.bar_label(container, fmt='%.2f')

# plt.tight_layout()
# plt.show()


# # SUS
# avg_trust = df.groupby(by)['SUS'].mean()
# plt.figure(figsize=(10, 6))
# bar_plot = avg_trust.plot(kind='bar', color='skyblue')
# plt.ylabel('Rata-rata')
# plt.title(f'Penilaian SUS Berdasarkan {by}')
# plt.xticks(rotation=0)  

# for i in bar_plot.containers:
#     bar_plot.bar_label(i, fmt='%.2f', label_type='edge')

# plt.tight_layout()
# plt.show()


# NPS
def calculate_nps(scores):
    promoters = (scores >= 9).sum()
    detractors = (scores <= 6).sum()
    total_responses = len(scores)
    nps = ((promoters - detractors) / total_responses) * 100
    return nps

nps_per_program = df.groupby(by)['NPS'].apply(calculate_nps)

plt.figure(figsize=(10, 6))
bar_plot = nps_per_program.plot(kind='bar', color='skyblue')
plt.ylabel('NPS')
plt.title(f'Penilaian NPS Berdasarkan {by}')
plt.xticks(rotation=0)

for i in bar_plot.containers:
    bar_plot.bar_label(i, fmt='%.2f', label_type='edge')

plt.tight_layout()
plt.show()