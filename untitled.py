import argparse
import math
import numpy as np
import pandas as pd
import random
import sys
import time
from functools import wraps
from sklearn import linear_model, metrics, preprocessing, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from math import sqrt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def get_dataframe_as_float_from_csv(csv_filename):
    """
    Returns a dataframe loaded from CSV and parsed as floats.  The reason why
    we need the columns to be floats is because NaN can only be inserted into
    float columns.

    TODO: Should we use null instead?  Maybe that doesn't suffer from the NaN
    problem, where the column dtype needs to be a float.
    """
    data = pd.read_csv('/Users/shrutibanda/GitHub/6.830-project/datasets/regression/red.csv')

    for col in data:
        #print('Column type is %s' % data[col].dtype)
        if is_numerical(data, col):
            data[col] = data[col].astype(float)

    return data

def is_numerical(df, column_name):
    """
    Whether this column has only numerical data.
    """
    return df[column_name].dtypes.kind in (np.typecodes["AllInteger"] + np.typecodes["AllFloat"])


def get_training_and_test_datasets(df):
    """
    Splits data into training and test data sets.
    """
    copy_df = df.copy()

    # Shuffle.
    #copy_df.reindex(np.random.permutation(copy_df.index))

    # Use a 70-30 breakdown for training and test data sets.
    num_rows_f = float(copy_df.shape[0])
    training_set_num_rows = int(math.ceil(.7 * num_rows_f))
    test_set_num_rows = int(math.floor(.3 * num_rows_f))
    print >> sys.stderr, 'num_rows_f=%f, training=%d, test=%d' % (
        num_rows_f, training_set_num_rows, test_set_num_rows)

    training_df = copy_df.head(training_set_num_rows)
    test_df = copy_df.tail(test_set_num_rows)

    return training_df, test_df


def get_encoded_df(df):
    """
    Converts values in all columns from string to integer.
    *** This does not convert string to integers - This converts categorical values to one hot encoding. So this must be only applied 
    to cateogircal values and not to all columns. 
    """
    le = LabelEncoder()
    for col in df.columns.values:
        #print('Column type is %s' % df[col].dtype)
        if(df[col].dtype == 'object'):
            print('Column being encoded' + col)
            data = df[col]
            le.fit(data.values)
            df[col] = le.transform(df[col])
    return df


if __name__ == "__main__":


    df = get_dataframe_as_float_from_csv('/Users/shrutibanda/GitHub/6.830-project/datasets/regression/red.csv')
    encoded_df = get_encoded_df(df)
    training_df, test_df = get_training_and_test_datasets(encoded_df)

    for column in training_df:
        print >> sys.stderr, '\nRunning ML for column \"%s\"' % column
        features = [c for c in training_df.columns if c is not column]

        y_train = training_df[column]
        X_train = training_df[features]
        y_test = test_df[column]
        X_test = test_df[features]

        lr = linear_model.LinearRegression()

        sfs = SFS(lr, 
          k_features=5, 
          forward=True, 
          floating=False, 
          scoring='neg_mean_squared_error',
          cv=10)
        sfs = sfs.fit(X_train, y_train)

        X_train = sfs1.transform(X_train)
        X_test = sfs1.transform(X_test)

        lr.fit(X_train_sfs, y_train)
        y_pred = lr.predict(X_test_sfs)
        # Compute the accuracy of the prediction
        acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
        print('Test set accuracy: %.2f %%' % (acc*100))

