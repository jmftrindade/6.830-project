import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns  # For prettier plots.
from scipy.cluster import hierarchy as hc
import numpy as np

# TODO: Generalize this script.
df = pd.read_csv('wine_dataset/red.csv')

# Per column stats (numerical columns only).
print df.describe(include='all').transpose()

# Histograms for all columns.
df.hist()

# Correlation dendrogram.
corr = 1 - df.corr()
corr_condensed = hc.distance.squareform(corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(20,12))
dendrogram = hc.dendrogram(z, labels=corr.columns,
                           link_color_func=lambda c: 'black')
plt.show()

# Use this for datasets with missing values only.
# msno.matrix(df)
# msno.dendrogram(df)
