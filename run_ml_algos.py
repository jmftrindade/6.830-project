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

        #print >> sys.stderr, stats_csv_header
       # print >> sys.stderr, stats_csv_data

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
        #print('Column type is %s' % data[col].dtype)
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


def run_all_classifiers(y_train, X_train, y_test, X_test, fn_stats_to_record,
                        fn_stats_to_record_from_result):
    # All classifiers that we consider.
    # TODO: Optionally investigate using GridSearch if the accuracies end up
    # being too low.
    classifiers = [
        {'name': 'Logistic Regression',
         'clf': linear_model.LogisticRegression()},
        {'name': 'Decision Tree',
         'clf': DecisionTreeClassifier(max_depth=1024, random_state=42)},
        {'name': 'SVC',
         'clf': svm.SVC()},
        {'name': 'SVMLinear',
         #'dt': svm.SVC(decision_function_shape='ovo')
         'clf': svm.SVC(kernel='linear')
         },
        {'name': 'Random Forest Classifier',
         'clf': RandomForestClassifier(max_depth=1024, random_state=42)}
    ]

    for classifier in classifiers:
        fn_stats_to_record['algo'] = classifier['name']
        res = run_classifier(
            y_train, X_train, y_test, X_test, classifier['clf'],
            additional_timer_stats=fn_stats_to_record,
            additional_timer_stats_from_result=fn_stats_to_record_from_result)
        print('%s classifier accuracy = %f %f' % (
            classifier['name'], res['training_accuracy'], res['test_accuracy']))


def run_all_regressors(y_train, X_train, y_test, X_test, fn_stats_to_record,
                       fn_stats_to_record_from_result):
    
    regressors = [
        {'name': 'Random Forest Regression',
         'regressor': RandomForestRegressor(n_estimators=15)},
        {'name': 'SVR',
         'regressor': svm.SVR(kernel='rbf', C=1e3, gamma=0.1)},
         {'name': 'Linear Regression',
         'regressor': linear_model.LinearRegression()}
    ]

    for regressor in regressors:
        #print('running regression now %s' + regressor['name'])
        model = regressor['regressor']
        print(model)
        sfs = SFS(model,k_features=8,forward=True,floating=False, 
            scoring='mean_squared_error',cv=2)
        sfs1 = sfs.fit(X_train, y_train)
        print('Selected features:', sfs1.k_feature_idx_)
        X_train = sfs1.transform(X_train)
        X_test = sfs1.transform(X_test)
        fn_stats_to_record['algo'] = regressor['name']
        res = run_regressor(
            y_train, X_train, y_test, X_test, 
            regressor['regressor'],
            additional_timer_stats=fn_stats_to_record,
            additional_timer_stats_from_result=fn_stats_to_record_from_result)
        print('%s Regressor MSE = %f %f Confidence %f' % (
            regressor['name'], res['training_mse'], res['test_mse'], res['confidence']))


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


def cross_validate(y_train, X_train, num_folds, ml_algo):
    # Shuffle cross-validation.
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    for train, test in k_fold.split(X_train):
        ml_algo.fit(X_train.iloc[train], y_train[train]).score(
            X_train.iloc[test], y_train[test])

    return ml_algo


@fn_timer
def run_regressor(y_train, X_train, y_test, X_test, regressor, *args, **kwargs):
    #print(regressor)

    #print(sfs1.subsets_)

    regressor = cross_validate(y_train, X_train, 5, regressor)

    training_mse = metrics.mean_squared_error(
        y_train, regressor.predict(X_train))
    test_mse = metrics.mean_squared_error(y_test, regressor.predict(X_test))
    confidence = metrics.r2_score(y_test, regressor.predict(X_test))

    # Organized as named entries in a dict for stats collection
    return {
        'test_accuracy': '',
        'training_accuracy': '',
        'test_mse': test_mse,
        'training_mse': training_mse,
        'confidence': confidence
    }


@fn_timer
def run_classifier(y_train, X_train, y_test, X_test, clf, *args, **kwargs):
    # 5-fold shuffle cross-validation.
    clf = cross_validate(y_train, X_train, 5, clf)

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
        'training_accuracy': training_accuracy,
        'test_mse': '',
        'training_mse': '',
        'confidence': ''
    }


def run_ml_for_all_columns(df):
    # FIXME: I don't think we can figure out an arbitrary "threshold" for this.
    # We should just run both classification and regression for numerical
    # columns, and record how many unique values we have.
    #threshold = 30

    # Only the columns that are categorical MUST be encoded. Not all the columns in a dataset. 

    encoded_df = get_encoded_df(df)
    training_df, test_df = get_training_and_test_datasets(encoded_df)

    for column in training_df:
        print >> sys.stderr, '\nRunning ML for column \"%s\"' % column
        features = [c for c in training_df.columns if c is not column]

        y_train = training_df[column]
        X_train = training_df[features]
        y_test = test_df[column]
        X_test = test_df[features]

        # Experiment stats to record per classifier run for numerical columns.
        fn_stats_to_record_from_result = ['test_accuracy', 'training_accuracy',
                                          'test_mse', 'training_mse']
        fn_stats_to_record = {
            'num_rows': training_df.shape[0],
            'target_num_unique': len(y_train.unique()),
            'target_variance': '',
            'target_stdev': ''
        }
        # Use both classification and regression for numerical data.
        if is_numerical(training_df, column):
            print('Column is numerical'+ column)
            scaled_df = get_scaled_dataframe(training_df)
            scaled_target = scaled_df[column]
            fn_stats_to_record['target_variance'] = scaled_target.var()
            fn_stats_to_record['target_stdev'] = scaled_target.std()
            run_all_regressors(y_train, X_train, y_test, X_test, fn_stats_to_record,
                               fn_stats_to_record_from_result)

        #run_all_classifiers(y_train, X_train, y_test, X_test, fn_stats_to_record,
                            #fn_stats_to_record_from_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Runs classification and regression algos over an input dataset')
    parser.add_argument(
        '-f',
        '--input_csv_file',
        help='Relative path of input CSV file containing data set with numeric'
        ' columns.',
        required='True')
    args = parser.parse_args()

    print('Running ML algos for input dataset \"%s\"' % args.input_csv_file)

    df = get_dataframe_as_float_from_csv(args.input_csv_file)

    #encoded_df = get_encoded_df(df)
    run_ml_for_all_columns(df)
