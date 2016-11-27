import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_plot(dataframe, xlabel, ylabel, output_figure=None, plot_kind=None):
    # Force a bar plot when there is only a single data point.
    if dataframe.shape[0] == 1:
        plot_kind = 'bar'

    plt.figure()
    ax = dataframe.plot(kind=plot_kind)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    if output_figure:
        plt.savefig(output_figure, format='pdf', bbox_inches='tight')
    else:
        plt.show()


def load_dataframe_from_csv(csv_filename, x_axis_column, y_axis_column, quantiles=[.99, .95, .50]):
    """
       Returns a merged Pandas dataframe with one column per requested quantile.
    """
    df = pd.read_csv(csv_filename)

    # Get only the X and Y columns.
    df = df[[y_axis_column, x_axis_column]].dropna()

    grouped_data = df.groupby(x_axis_column)
    #print(grouped_data.head())

    quantile_dfs = []
    for quantile in quantiles:
        quantile_df = grouped_data.quantile(quantile)[[y_axis_column]]
        #print('\nquantile_df for quantile %s' % str(quantile))
        #print(quantile_df)

        quantile_df.columns = [str(int(quantile * 100)) + 'th %ile']
        #print('\nquantile_df.columns:')
        #print(quantile_df.columns)

        quantile_dfs.append(quantile_df)
        #print(quantile_df.head())

    all_quantiles_df = None
    if len(quantile_dfs) > 0:
        index_for_join = quantile_dfs[0].index
        all_quantiles_df = pd.concat(quantile_dfs, axis=1, join_axes=[
                                     quantile_dfs[0].index])
    return all_quantiles_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots a line chart of the average values for a given group in a csv file.")
    parser.add_argument('-i', '--input_csv',
                        help='Relative path of input CSV filename.',
                        required=True)
    parser.add_argument('-o', '--output_figure',
                        help='Relative path of optional output figure PDF filename.')
    parser.add_argument('-x', '--x_axis_column',
                        help='Name of column in CSV file to plot as x axis, e.g., COLUMN_LENGTH.')
    parser.add_argument('-y', '--y_axis_column',
                        help='Name of column in CSV file to plot as y axis, e.g., TIME.')
    parser.add_argument('-pk', '--plot_kind',
                        help='Optional plot kind, e.g., "bar", "line" etc.  Defaults to "line".')
    args = parser.parse_args()

    column_length_percentiles_df = load_dataframe_from_csv(
        args.input_csv, args.x_axis_column, args.y_axis_column)
    generate_plot(column_length_percentiles_df,
                  xlabel=args.x_axis_column,
                  ylabel=args.y_axis_column,
                  output_figure=args.output_figure,
                  plot_kind=args.plot_kind)
