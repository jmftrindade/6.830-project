# TODO: Update this with the final report.

# 6.830-project
Class project for 6.830 database systems.

# Generate Missing Value training and test data sets:

The script ```generate_missing_value_data_sets.py``` can be used as follows:

```
$ python generate_missing_value_datasets.py -h
usage: generate_missing_value_datasets.py [-h] -f INPUT_CSV_FILE
                                          -m MISSING_VALUE_FRACTION

Generate missing values training and test data sets.

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_CSV_FILE, --input_csv_file INPUT_CSV_FILE
                        Relative path of input CSV file containing data set
                        with numeric columns.
  -m MISSING_VALUE_FRACTION, --missing_value_fraction MISSING_VALUE_FRACTION
                        Fraction of missing values to be inserted.
```

E.g.,

```
$ python generate_missing_value_datasets.py -f wine_dataset/red.csv -m 0.5
```

which will generate training and test data set files under
```wine_dataset/red_training.csv``` and ```wine_dataset/red_test.csv```
respectively.  The training data set contains the first 70% rows, and target
columns, while the test data set contains the last 70% rows of the original
dataset, and no target columns.
