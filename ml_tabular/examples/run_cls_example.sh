python3 -u ./autotabular/classification/train.py \
            --train_input alita_dev.iris_3class \
            --output_dir tmp/tabular_test.outputs \
            --data_format table \
            --label_cols "['label']" \
            --feature_cols "['sepal_length','sepal_width','petal_length','petal_width']" \
            --time_budget 30 \
            --metric accuracy \
            --estimator_list '["lgbm","xgboost"]' \
            --seed 1
