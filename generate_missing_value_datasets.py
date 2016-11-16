import argparse
import math
import numpy as np
import os
import pandas as pd
import random


def get_dataframe_as_float_from_csv(csv_filename):
    """
    Returns a dataframe loaded from CSV and parsed as floats.  The reason why
    we need the columns to be floats is because NaN can only be inserted into
    float columns.

    TODO: Should we use null instead?  Maybe that doesn't suffer from the NaN
    problem, where the column dtype needs to be a float.
    """
    data = pd.read_csv(csv_filename)

    for col in data:
        if data[col].dtypes == 'int64':
            data[col] = data[col].astype(float)

    return data


def get_df_with_missing_value_MCAR(df, sample_fraction):
    """
    Inserts missing values MCAR (Missing Completely At Random) into the input
    data set, using the specified sample fraction.

    Note that if the input dataframe already contains NaN, then the resulting
    fraction of NaN may be larger than the specified sample_fraction.
    """
    if sample_fraction < 0.0 or sample_fraction > 1.0:
        raise ValueError("sample fraction must be number between 0 and 1")

    # Copy into a separate dataframe.  Note that if the original dataframe
    # is prohibitively large, this call is doubling that size.
    df_with_nan = df.copy()

    # Create an array where every tuple is the row and column number.
    ix = [(row, col) for row in range(df_with_nan.shape[0])
          for col in range(df_with_nan.shape[1])]

    # Then for each (row, col) in the random sample, insert a NaN on that cell.
    for row, col in random.sample(ix, int(round(sample_fraction * len(ix)))):
      df_with_nan.iat[row, col] = np.nan

    return df_with_nan


def get_target_df_column_names(df):
    """
    Returns an array of column names that are renamed versions of the column
    names from the input dataframe.
    """
    target_df_column_names = []
    for column in df:
        target_df_column_names.append(column + '_target')
    return target_df_column_names


def generate_training_and_test_datasets(csv_filename, sample_fraction):
    """
    Returns a dataframe concatenating the dataframe with inserted missing
    values, and the target variables dataframe (essentially the original data).
    """
    target_df = get_dataframe_as_float_from_csv(csv_filename)
    df_with_nan = get_df_with_missing_value_MCAR(target_df, sample_fraction)
    target_df.columns = get_target_df_column_names(target_df)

    # TODO: Figure out whether we'll use this tuple column technique in any way.
    #
    # concatenated = pd.concat([df_with_nan, target_df], axis=1)
    # concatenated['target'] = concatenated[target_df_column_names].apply(tuple, axis=1)
    #
    training_df = pd.concat([df_with_nan, target_df], axis=1)

    # Use a 70-30 breakdown for training and test data sets, where upper 70%
    # are training, and bottom 30% rows are test.
    #
    # FIXME do we need to shuffle the rows first?
    training_set_num_rows = int(math.ceil(.7 * float(training_df.shape[0])))
    test_set_num_rows = int(math.floor(.3 * float(training_df.shape[0])))

    output_filename_prefix = os.path.splitext(csv_filename)
    training_set_filename = output_filename_prefix[0] + "_training.csv"
    test_set_filename = output_filename_prefix[0] + "_test.csv"

    training_df.head(training_set_num_rows).to_csv(
        training_set_filename,
        index=False)
    df_with_nan.tail(test_set_num_rows).to_csv(
        test_set_filename,
        index=False)

    print('Saved training data set under \"%s\"' % training_set_filename)
    print('Saved test data set under \"%s\"' % test_set_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate missing values training and test data sets.')
    parser.add_argument(
        '-f',
        '--input_csv_file',
        help='Relative path of input CSV file containing data set with numeric'
        ' columns.',
        required='True')
    parser.add_argument(
        '-m',
        '--missing_value_fraction',
        type=float,
        help='Fraction of missing values to be inserted.',
        required='True')
    args = parser.parse_args()

    print('Generating missing value training and test data sets from \"%s\"'
          % args.input_csv_file)
    generate_training_and_test_datasets(
        args.input_csv_file,
        args.missing_value_fraction)
