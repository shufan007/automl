
#export CUDA_VISIBLE_DEVICES=1

#cd ..

# ETT h1
python -u forecasting/run.py \
  --seed 1234 \
  --time_budget_s 300 \
  --num_samples 4 \
  --num_gpus_per_worker 1 \
  --hadoop_config '{"HADOOP_USER_NAME":"prod_alita","HADOOP_USER_PASSWORD":"xxx"}' \
  --data_format table \
  --train_data alita_dev.etth1 \
  --split_ratio 0.2 \
  --output_path tmp/models/ts_model \
  --search_space '{"model":["FEDformer","Autoformer"],"d_model":[64,512],"n_heads":[4,8],"d_ffn":[128,2048],"learning_rate":[0.00001,0.001]}' \
  --features_col "['hufl','hull','mufl','mull','lufl','lull']" \
  --label_col ot \
  --date_col date \
  --freq h \
  --seq_len 96 \
  --pred_len 24 \
  --moving_avg 25 \
  --train_epochs 2 \
  --patience 2 \
  --batch_size 32 \
  --loss mse \
  --sample_step 16 \
