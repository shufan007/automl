
#export CUDA_VISIBLE_DEVICES=1

#cd ..

# ETT h1
python -u forecasting/run.py \
  --seed 1234 \
  --time_budget_s 300 \
  --num_samples 4 \
  --num_gpus_per_worker 1 \
  --hadoop_config '{"HADOOP_USER_NAME":"prod_alita","HADOOP_USER_PASSWORD":"xxx"}' \
  --data_format array-table \
  --train_data alita_dev.etth1_seq8_pred4_1w \
  --split_ratio 0.2 \
  --output_path tmp/models/ts_model \
  --search_space '{"model":["FEDformer","Autoformer"],"d_model":[64,512],"n_heads":[4,8],"d_ffn":[128,2048],"learning_rate":[0.00001,0.001]}' \
  --label_col target \
  --features_col "['features']" \
  --moving_avg 3 \
  --train_epochs 2 \
  --batch_size 32 \
  --loss mse \
  --sample_step 16 \
