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
        if is_numerical(data, col):
            data[col] = data[col].astype(float)

    return data


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


def get_scaled_dataframe(df):
    """
    Produces a scaled dataframe that we can use to compute aggregate stats
    (measures of central tendency and dispersion).
    """
    x = df.values  # The underlying numpy array.
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)


def is_numerical(df, column_name):
    """
    Whether this column has only numerical data.
    """
    return df[column_name].dtypes.kind in (np.typecodes["AllInteger"] + np.typecodes["AllFloat"])


def run_all_classifiers(target, features, training_df, test_df):
    """Try all classification ML algorithms and report their accuracies."""

    # All classifiers that we consider.
    # TODO: Optionally investigate using GridSearch if the accuracies end up
    # being too low.
    classifiers = [
        {'name': 'Logistic Regression',
         'clf': linear_model.LogisticRegression()},
        {'name': 'Decision Tree',
         'clf': DecisionTreeClassifier(max_depth=1024, random_state=42)},
        {'name': 'SVM',
         'clf': svm.SVC()},
        {'name': 'Random Forest Classifier',
         'clf': RandomForestClassifier(max_depth=1024, random_state=42)}
    ]

    y_train = training_df[target]
    X_train = training_df[features]
    y_test = test_df[target]
    X_test = test_df[features]

    # Experiment stats to record per classifier run for numerical columns.
    stdev = None
    var = None
    if is_numerical(training_df, target):
        scaled_df = get_scaled_dataframe(training_df)
        scaled_target = scaled_df[target]
        var, stdev = scaled_target.var(), scaled_target.std()
    fn_stats_to_record = {
        'num_rows': training_df.shape[0],
        'target_num_unique': len(y_train.unique()),
        'target_variance': var if var is not None else '',
        'target_stdev': stdev if stdev is not None else ''
    }
    fn_stats_to_record_from_result = ['test_accuracy', 'training_accuracy']

    for classifier in classifiers:
        fn_stats_to_record['algo'] = classifier['name']
        res = run_classifier(y_train, X_train, y_test, X_test,
                             classifier['clf'],
                             additional_timer_stats=fn_stats_to_record,
                             additional_timer_stats_from_result=fn_stats_to_record_from_result)
        #print('%s classifier accuracy = %f' % (
        #    classifier['name'], res['test_accuracy']))


def get_encoded_df(df):
    """
    Converts values in all columns from string to integer.
    """
    le = LabelEncoder()
    for col in df.columns.values:
        data = df[col]
        le.fit(data.values)
        df[col] = le.transform(df[col])
    return df


@fn_timer
def run_classifier(y_train, X_train, y_test, X_test, clf, *args, **kwargs):
    # 5-fold shuffle cross-validation.
    k_fold = KFold(n_splits=5, shuffle=True)
    for train, test in k_fold.split(X_train):
        clf.fit(X_train.iloc[train], y_train[train]).score(
            X_train.iloc[test], y_train[test])

    # Train calibrated classifier with 5-fold cross-validation.
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_train, y_train)  # Not needed because clf is prefit.

    # Accuracy over training data set.
    training_accuracy = metrics.accuracy_score(
        y_train, calibrated_clf.predict(X_train))  # clf.predict(X_train))
    # Accuracy over test data set.
    test_accuracy = metrics.accuracy_score(
        y_test, calibrated_clf.predict(X_test))  # clf.predict(X_test))

    # Organized as named entries in a dict for stats collection.
    return {
        'test_accuracy': test_accuracy,
        'training_accuracy': training_accuracy
    }


def run_ml_for_all_columns(df):
    # FIXME: I don't think we can figure out an arbitrary "threshold" for this.
    # We should just run both classification and regression for numerical
    # columns, and record how many unique values we have.
    #threshold = 30

    encoded_df = get_encoded_df(df)
    training_df, test_df = get_training_and_test_datasets(encoded_df)

    for column in training_df:
        print >> sys.stderr, '\nRunning ML for column \"%s\"' % column
        features = [c for c in training_df.columns if c is not column]

        # Use both classification and regression for numerical data.
        # All numerical data is read as floats, so no need to check for other
        # numerical data types here.
        if is_numerical(training_df, column):
            print >> sys.stderr, 'TODO: check accuracy for regression.'

        run_all_classifiers(column, features, training_df, test_df)


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
