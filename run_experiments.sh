#!/usr/local/bin/bash
#
# Requires Bash v4 for associative arrays.
# Mac install with Homebrew will make it available under /usr/local/bin.

# Globals.
TODAY=`date "+%Y%m%d"`
NUMBER_RUNS=1  #10

# Experiment -> dataset directory
declare -A EXPERIMENT_DATASETS
EXPERIMENT_DATASETS["categorical"]=datasets/classification/*csv
#EXPERIMENT_DATASETS["numerical"]=datasets/regression/*csv
#EXPERIMENT_DATASETS["FD_paper"]=datasets/from_FD_paper/*csv

# Run experiment and generate logs.
function run_experiment {
  local input_dataset=$1
  local output_log_file=$2
  local number_runs=$3

  echo "input_dataset=$input_dataset"
  echo "output_log_file=$output_log_file"
  echo "number_runs=$number_runs"

  args="-m timeit -n 1 -r ${number_runs} 'import os' 'os.system(\"python run_ml_algos.py -f ${input_dataset}\")' > ${output_log_file} 2>&1"
  echo "Running:"
  echo "python $args"
  eval python $args
  echo "Saved logs as \"${output_log_file}\"."
}

# Generate CSV from experiment logs.
function generate_csv {
  local input_log_file=$1
  local output_csv_file=$2

  # Generate CSV from experiment logs.
  echo "Generating CSV..."
  grep '^run_classifier\|^run_regressor' ${input_log_file} | sed -e $'1i\\\nfunction,runtime_seconds,target_stdev,target_variance,target_num_unique,num_rows,target_is_numerical,algo,test_accuracy,training_accuracy,test_mse,training_mse' > ${output_csv_file}
  echo "Saved CSV as \"${output_csv_file}\"."
}

# Generates plot from CSV.
function plot_csv {
  local input_csv_file=$1
  local output_plot_file=$2
  local x_axis_column=$3
  local y_axis_column=$4
  local plot_kind=${5-line}  # Default to "line" if not available.

  echo "Plotting results..."
  echo "python plot_experiment_results.py -i ${input_csv_file} -o ${output_plot_file} -x ${x_axis_column} -y ${y_axis_column} -pk ${plot_kind}"
  python plot_experiment_results.py -i ${input_csv_file} -o ${output_plot_file} -x ${x_axis_column} -y ${y_axis_column} -pk ${plot_kind}
  echo "Saved plot as \"${output_plot_file}\"."
}

for experiment in "${!EXPERIMENT_DATASETS[@]}";
do
  echo "experiment=$experiment"

  input_datasets_dir=${EXPERIMENT_DATASETS[$experiment]}

  for input_dataset_file in ${input_datasets_dir} ;
  do
    file_basename=$(basename $input_dataset_file)
    file_basename_no_extension=$TODAY-$experiment-${file_basename%.*}
    log_filename=experiment_logs/${file_basename_no_extension}.log
    csv_filename=experiment_logs/${file_basename_no_extension}.csv
    plot_file_basename=plots/${file_basename_no_extension}

    run_experiment $input_dataset_file $log_filename $NUMBER_RUNS
    generate_csv $log_filename $csv_filename

    # TODO: Generalize this code.
    # X axis = ML algorithm bar plots.
    plot_csv $csv_filename $plot_file_basename-algo_vs_runtime.pdf algo runtime_seconds bar
    plot_csv $csv_filename $plot_file_basename-algo_vs_test_accuracy.pdf algo test_accuracy bar
    plot_csv $csv_filename $plot_file_basename-algo_vs_training_accuracy.pdf algo training_accuracy bar
    plot_csv $csv_filename $plot_file_basename-algo_vs_test_mse.pdf algo test_mse bar
    plot_csv $csv_filename $plot_file_basename-algo_vs_training_mse.pdf algo training_mse bar
    # X axis = number of uniques in target variable.
    plot_csv $csv_filename $plot_file_basename-num_unique_vs_runtime.pdf target_num_unique runtime_seconds
    plot_csv $csv_filename $plot_file_basename-num_unique_vs_test_accuracy.pdf target_num_unique test_accuracy
    plot_csv $csv_filename $plot_file_basename-num_unique_vs_training_accuracy.pdf target_num_unique training_accuracy
    plot_csv $csv_filename $plot_file_basename-num_unique_vs_test_mse.pdf target_num_unique test_mse
    plot_csv $csv_filename $plot_file_basename-num_unique_vs_training_mse.pdf target_num_unique training_mse
    # X axis = target variable scaled and normalized variance.
    plot_csv $csv_filename $plot_file_basename-variance_vs_training_mse.pdf target_variance training_mse
    plot_csv $csv_filename $plot_file_basename-variance_vs_test_mse.pdf target_variance test_mse
    # X axis = target variable scaled and normalized standard deviation.
    plot_csv $csv_filename $plot_file_basename-stdev_vs_training_mse.pdf target_stdev training_mse
    plot_csv $csv_filename $plot_file_basename-stdev_vs_test_mse.pdf target_stdev test_mse
  done

  # Merge all logs so we can plot data from multiple data sets.
  all_datasets_logs=experiment_logs/$TODAY-$experiment-*.log
  # Because of this, we should not have a dataset called "all.csv" :-)
  all_datasets_csv_filename=experiment_logs/$TODAY-$experiment-all.csv
  all_datasets_plot_file_basename=plots/$TODAY-$experiment-all

  generate_csv $all_datasets_logs $all_datasets_csv_filename

  # It only makes sense to plot these for multiple data sets.
  # X axis = number of rows.
  plot_csv $all_datasets_csv_filename $all_datasets_plot_file_basename-num_rows_vs_runtime.pdf num_rows runtime_seconds
  plot_csv $all_datasets_csv_filename $all_datasets_plot_file_basename-num_rows_vs_test_accuracy.pdf num_rows test_accuracy
  plot_csv $all_datasets_csv_filename $all_datasets_plot_file_basename-num_rows_vs_training_accuracy.pdf num_rows training_accuracy
done
