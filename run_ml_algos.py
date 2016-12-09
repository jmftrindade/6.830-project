import argparse
import math
import numpy as np
import pandas as pd
import random
import sys
import time

# Supress the annoying LAPACK gelsd warning.
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

from functools import wraps
from math import sqrt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
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


def read_dataframe_without_na_from_csv(csv_filename):
    """
    Returns a dataframe loaded from CSV where rows with at least 1 NaN were
    dropped.
    """
    df = pd.read_csv(csv_filename)

    # Drop any rows that contain at least one NaN.
    df_no_na = df.dropna().reset_index(drop=True)
    num_dropped = len(df) - len(df_no_na)
    if num_dropped > 0:
        print >> sys.stderr, 'WARNING: dropped %d NaN rows.' % (num_dropped)

    return df_no_na


# UNUSED: only necessary when we start running experiments with missing data.
# And we should probably reconsider this approach for the NaN, given that once
# we convert the column from int to float, we lose the info that it was a
# originally numerical discrete column (instead of continuous).
def get_dataframe_as_float_from_csv(csv_filename):
    """
    Returns a dataframe loaded from CSV and parsed as floats.  The reason why
    we need the columns to be floats is because NaN can only be inserted into
    float columns.
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
    Whether this column has only numerical data, which can be either discrete
    or continuous.
    """
    return df[column_name].dtypes.kind in (
        np.typecodes["AllInteger"] + np.typecodes["AllFloat"])


def is_continuous(df, column_name):
    """
    Whether this column has only continuous numerical data.
    """
    return df[column_name].dtypes.kind in (np.typecodes["AllFloat"])


def run_all_classifiers(y_train, X_train, y_test, X_test, fn_stats_to_record,
                        fn_stats_to_record_from_result):
    # All classifiers that we consider.
    # TODO: Optionally investigate using GridSearch if the accuracies end up
    # being too low.
    # TODO: Consider using n_jobs=-1 for the classifiers that accept it when
    # we run these experiments on a machine that has more than just 1 cpu.
    classifiers = [
        {'name': 'LogRC',
         'clf': linear_model.LogisticRegression(n_jobs=-1)},
        {'name': 'DTC',
         'clf': DecisionTreeClassifier(max_depth=1024, random_state=42)},
# SVM classifiers are too slow.
#        {'name': 'SVC',
#         'clf': svm.SVC()},
#        {'name': 'LinSVC',
#         'clf': svm.SVC(kernel='linear')},
        {'name': 'RFC',
         'clf': RandomForestClassifier(max_depth=1024, random_state=42, n_jobs=-1)}
    ]

    for classifier in classifiers:
        fn_stats_to_record['algo'] = classifier['name']
        res = run_classifier(
            y_train, X_train, y_test, X_test, classifier['clf'],
            additional_timer_stats=fn_stats_to_record,
            additional_timer_stats_from_result=fn_stats_to_record_from_result)


def run_all_regressors(y_train, X_train, y_test, X_test, fn_stats_to_record,
                       fn_stats_to_record_from_result):
    regressors = [
        {'name': 'RFR',
         'regressor': RandomForestRegressor(n_estimators=15, n_jobs=-1)},
# SVM Regressors are too slow.
#        {'name': 'SVR',
#         'regressor': svm.SVR()},
#        {'name': 'CstmSVR',  # Custom SVR for comparison.
#         'regressor': svm.SVR(kernel='rbf', C=1e3, gamma=0.1)},
        {'name': 'LinR',
         'regressor': linear_model.LinearRegression(n_jobs=-1)}
    ]

    for regressor in regressors:
        fn_stats_to_record['algo'] = regressor['name']
        res = run_regressor(
            y_train, X_train, y_test, X_test, regressor['regressor'],
            additional_timer_stats=fn_stats_to_record,
            additional_timer_stats_from_result=fn_stats_to_record_from_result)


def get_encoded_df(df):
    """
    Converts values in string-based categorical feature columns to one hot
    encoding.

    Operates over a copy of the original dataframe, so that the original data
    is not modified.  Only applies label encoding to columns that are not
    numerical.  We limit label encoding to non-numerical columns because
    otherwise numbers with float or double precision would be converted to
    integers, which loses the original precision.  And in the case of integers
    it would also incorrectly change the range if, for example, it was a sparse
    column.
    """
    copy_df = df.copy()
    le = LabelEncoder()
    for col in copy_df.columns.values:
        if not is_numerical(copy_df, col):
            data = copy_df[col]
            le.fit(data.values)
            copy_df[col] = le.transform(copy_df[col])
    return copy_df


def cross_validate(y_train, X_train, num_folds, ml_algo):
    # Shuffle cross-validation.
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    for train, test in k_fold.split(X_train):
        # NOTE: This is used if using pandas objects.
        ml_algo.fit(X_train.iloc[train], y_train[train]).score(
            X_train.iloc[test], y_train[test])
        # NOTE: This is used if using raw numpy arrays, which is the case with
        # SFS in run_regressor and run_classifier.
        # ml_algo.fit(X_train[train], y_train[train]).score(
        #    X_train[test], y_train[test])

    return ml_algo


@fn_timer
def run_regressor(y_train, X_train, y_test, X_test, regressor, *args, **kwargs):
    # Sequential feature selection with cross-validation. We limit SFS to
    # search only for up to ceil[num_columns / 2], as that should suffice
    # to find a decent combination of features that predicts the target
    # variable.
#    print >> sys.stderr, 'Running SFS:'

    # 5-fold cross-validation if we have at least 100 instances on training
    # data, and 2-fold otherwise.
    cv_k = 5 if len(y_train) > 100 else 2

    # TODO: Enable n_jobs=-1 to take advantage of all CPUs available.
#    sfs = SFS(regressor,
#              k_features=(1, int(math.ceil(len(X_train.columns) / 2))),
#              forward=True,
#              floating=False,
#              scoring='neg_mean_squared_error',
#              print_progress=False,
#              cv=cv_k)
#    # The mlxtend library's SFS expects underlying numpy array (as_matrix()).
#    sfs = sfs.fit(X_train.as_matrix(), y_train)
#
#    print 'SFS features and scores: %s' % sfs.subsets_
#
#    # Use SFS results to improve the model.
#    X_train_sfs = sfs.transform(X_train.as_matrix())
#    X_test_sfs = sfs.transform(X_test.as_matrix())

    # Do cross-validation for the fit of transformed SFS features.
#    regressor = cross_validate(y_train, X_train_sfs, cv_k, regressor)
    regressor = cross_validate(y_train, X_train, cv_k, regressor)

#    training_mse = metrics.mean_squared_error(
#        y_train, regressor.predict(X_train_sfs))
#    test_mse = metrics.mean_squared_error(
#        y_test, regressor.predict(X_test_sfs))
#    r2_score = metrics.r2_score(y_test, regressor.predict(X_test_sfs))
    training_mse = metrics.mean_squared_error(
        y_train, regressor.predict(X_train))
    test_mse = metrics.mean_squared_error(
        y_test, regressor.predict(X_test))
    r2_score = metrics.r2_score(y_test, regressor.predict(X_test))

    # Organized as named entries in a dict for stats collection
    return {
        'test_accuracy': '',
        'training_accuracy': '',
        'test_mse': test_mse,
        'training_mse': training_mse,
        'test_R2_score': r2_score
    }


@fn_timer
def run_classifier(y_train, X_train, y_test, X_test, clf, *args, **kwargs):
    # Sequential feature selection with cross-validation. We limit SFS to
    # search only for up to ceil[num_columns / 2], as that should suffice
    # to find a decent combination of features that predicts the target
    # variable.
#    print >> sys.stderr, 'Running SFS:'

    # 5-fold cross-validation if we have at least 100 instances on training
    # data, and 2-fold otherwise.
    cv_k = 5 if len(y_train) > 100 else 2

    # TODO: Enable n_jobs=-1 to take advantage of all CPUs available.
#    sfs = SFS(clf,
#              k_features=(1, int(math.ceil(len(X_train.columns) / 2))),
#              forward=True,
#              floating=False,
#              scoring='accuracy',
#              print_progress=False,
#              cv=cv_k)
#
#    # The mlxtend library's SFS expects underlying numpy array (as_matrix()).
#    sfs = sfs.fit(X_train.as_matrix(), y_train)

#    print 'SFS features and scores: %s' % sfs.subsets_

    # Use SFS results to improve the model.
#    X_train_sfs = sfs.transform(X_train.as_matrix())
#    X_test_sfs = sfs.transform(X_test.as_matrix())

    # Do cross-validation for the fit of transformed SFS features.
#    clf = cross_validate(y_train, X_train_sfs, cv_k, clf)
    clf = cross_validate(y_train, X_train, cv_k, clf)

    # Train calibrated classifier with prefit cross-validation, as CV is
    # performed above.
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
#    calibrated_clf.fit(X_train_sfs, y_train)
    calibrated_clf.fit(X_train, y_train)

    # Accuracy over training data set.
#    training_accuracy = metrics.accuracy_score(
#        y_train, calibrated_clf.predict(X_train_sfs))
#    # Accuracy over test data set.
#    test_accuracy = metrics.accuracy_score(
#        y_test, calibrated_clf.predict(X_test_sfs))

    # Accuracy over training data set.
    training_accuracy = metrics.accuracy_score(
        y_train, calibrated_clf.predict(X_train))
    # Accuracy over test data set.
    test_accuracy = metrics.accuracy_score(
        y_test, calibrated_clf.predict(X_test))

    # Organized as named entries in a dict for stats collection.
    return {
        'test_accuracy': test_accuracy,
        'training_accuracy': training_accuracy,
        'test_mse': '',
        'training_mse': '',
        'test_R2_score': ''
    }


def run_ml_for_all_columns(df):
    encoded_df = get_encoded_df(df)
    training_df, test_df = get_training_and_test_datasets(encoded_df)

    for column in training_df:
        # FIXME remove this hack: skipping that slow column.
        if column == 'fnlwgt':
            continue

        print >> sys.stderr, '\nRunning ML for column \"%s\"' % column
        features = [c for c in training_df.columns if c is not column]

        y_train = training_df[column]
        X_train = training_df[features]
        y_test = test_df[column]
        X_test = test_df[features]

        # Experiment stats to record per classifier run for numerical columns.
        fn_stats_to_record_from_result = ['test_accuracy', 'training_accuracy',
                                          'test_mse', 'training_mse']
        is_column_continuous = is_continuous(df, column)
        is_column_numerical = is_numerical(df, column)
        fn_stats_to_record = {
            'column_name': column,
            'num_rows': training_df.shape[0],
            'target_num_unique': len(y_train.unique()),
            'target_variance': '',
            'target_stdev': '',
            'target_is_numerical': is_numerical(df, column),
            'target_is_continuous': is_continuous(df, column)
        }

        # Record aggregate statistics for data that was originally numerical.
        if is_numerical(df, column):
            scaled_df = get_scaled_dataframe(training_df)
            scaled_target = scaled_df[column]
            fn_stats_to_record['target_variance'] = scaled_target.var()
            fn_stats_to_record['target_stdev'] = scaled_target.std()

        # Only use classification for discrete data (numerical or not).
        if not is_continuous(df, column):
            run_all_classifiers(y_train, X_train, y_test, X_test, fn_stats_to_record,
                                fn_stats_to_record_from_result)

        # While we use regression for any data set, since after encoding
        # categorical data, all the columns are numerical (discrete or
        # continuous).
        run_all_regressors(y_train, X_train, y_test, X_test, fn_stats_to_record,
                           fn_stats_to_record_from_result)


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

    df = read_dataframe_without_na_from_csv(args.input_csv_file)
    run_ml_for_all_columns(df)
