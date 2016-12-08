import argparse
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns  # For prettier plots.
from scipy.cluster import hierarchy as hc
import numpy as np


def plot_histogram_and_correlation_dendrogram(csv_filename):
    df = pd.read_csv(csv_filename)

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

    # Only use this for datasets with missing values.
    if df.isnull().values.any():
        msno.matrix(df)
        msno.dendrogram(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots histograms and correlation dendrogram for a given dataset.")
    parser.add_argument('-i', '--input_csv',
                        help='Relative path of input CSV filename.',
                        required=True)
    args = parser.parse_args()

    plot_histogram_and_correlation_dendrogram(args.input_csv)
