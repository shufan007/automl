
python3 -u ./classification/run.py \
--hadoop_config '{"HADOOP_USER_NAME":"prod_alita","HADOOP_USER_PASSWORD":"xxx"}' \
--data_format table \
--train_data alita_dev.iris_3class \
--label_col label \
--features_col "['sepal_length','sepal_width','petal_length','petal_width']" \
--output_path /nfs/volume-807-1/fanshuangxi/tmp/tabular_test.outputs \
--time_budget 30 \
--metric accuracy \
--estimator_list '["lgbm"]' \
--estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":3}' \
--seed 1
