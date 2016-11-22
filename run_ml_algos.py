import argparse
import numpy as np
import pandas as pd
import random
import sys
import time
from functools import wraps
from sklearn import linear_model, metrics, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def fn_timer(function):
    """
    Timer decorator that prints total running time of execution.

    A function decorated by this may also provide additional stats via
    a "additional_timer_stats" kwarg.
    """
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()

        # First column is time.
        stats_csv_header = 'function,runtime_seconds'
        stats_csv_data = '%s,%s' % (function.func_name, str(t1 - t0))

        # Remaining columns.
        if 'additional_timer_stats' in kwargs:
            for k, v in kwargs['additional_timer_stats'].iteritems():
                stats_csv_header += ',%s' % k
                stats_csv_data += ',%s' % str(v)

        if 'additional_timer_stats_from_result' in kwargs:
            for stat in kwargs['additional_timer_stats_from_result']:
                stats_csv_header += ',%s' % stat
                stats_csv_data += ',%s' % str(result[stat])

        print >> sys.stderr, stats_csv_header
        print >> sys.stderr, stats_csv_data

        return result
    return function_timer


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
        if isinstance(data[col].dtypes, (int)):
            data[col] = data[col].astype(float)

    return data


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    # FIXME: Is this replacing the values in the original column?
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


def run_all_classifiers(targ, features, df):
    """Try all classification ML algorithms and report their accuracies."""

    df2, targets = encode_target(df, targ)

    y = df2["Target"]
    X = df2[features]

    # All classifiers that we consider.
    # TODO: Optionally investigate using GridSearch if the accuracies end up
    # being too low.
    classifiers = [
        {'name': 'Logistic Regression',
         'dt': linear_model.LogisticRegression()},
        {'name': 'Decision Tree',
         'dt': DecisionTreeClassifier(min_samples_split=30, random_state=99)},
        {'name': 'SVM',
         'dt': svm.SVC(decision_function_shape='ova')}
    ]

    # Experiment stats to record per classifier run.
    fn_stats_to_record = {
        'target_num_unique': len(y.unique())
    }
    fn_stats_to_record_from_result = ['accuracy']

    for classifier in classifiers:
        fn_stats_to_record['algo'] = classifier['name']
        res = run_classifier(
            y, X, classifier['dt'],
            # NOTE: This is intentional, as these need to be kwargs.
            additional_timer_stats=fn_stats_to_record,
            additional_timer_stats_from_result=fn_stats_to_record_from_result)
        #print('%s classifier accuracy = %f' % (
        #    classifier['name'], res['accuracy']))


@fn_timer
def run_classifier(y, X, dt, *args, **kwargs):
    # Convert from string to integer.
    le = LabelEncoder()
    for col in X.columns.values:
        data = X[col]
        le.fit(data.values)
        X[col] = le.transform(X[col])
    dt.fit(X, y)
    expected = y
    predicted = dt.predict(X)

    # Organized as named entries in a dict for stats collection.
    return {'accuracy': metrics.accuracy_score(expected, predicted)}


def run_ml_for_all_columns(df):
    # FIXME: I don't think we can figure out an arbitrary "threshold" for this.
    # We should just run both classification and regression for numerical
    # columns, and record how many unique values we have.
    #threshold = 30

    for col in df:
        print('\nRunning ML for column \"%s\"' % col)
        features = [column for column in df.columns if column is not col]

        # Use both classification and regression for numerical data.
        # All numerical data is read as floats, so no need to check for other
        # numerical data types here.
        if df[col].dtypes == 'float64':
            print('TODO: check accuracy for regression.')

        run_all_classifiers(col, features, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs classification and regression algos over an input dataset.')
    parser.add_argument(
        '-f',
        '--input_csv_file',
        help='Relative path of input CSV file containing data set with numeric'
        ' columns.',
        required='True')
    args = parser.parse_args()

    print('Running ML algos for input dataset \"%s\"' % args.input_csv_file)

    df = get_dataframe_as_float_from_csv(args.input_csv_file)
    run_ml_for_all_columns(df)
