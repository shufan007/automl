# export CUDA_VISIBLE_DEVICES=1

#cd ..

# ETT h1
python -u autosearch/run.py \
  --seed 1234 \
  --task_type forecasting \
  --time_budget_s 120 \
  --num_samples 4 \
  --num_gpus_per_worker 1 \
  --resources_per_trial '{"cpu":1,"gpu":1}' \
  --search_space '{"model":{"htype":"choice","value":["FEDformer","Autoformer"]},"d_model":{"htype":"exponent","value":[64,512]},"n_heads":{"htype":"qrandint","value":[4,12,4]},"d_ffn":{"htype":"exponent","value":[128,2048]},"learning_rate":{"htype":"loguniform","value":[0.0001,0.01]}}' \
  --metric val_loss \
  --metric_mode min \
  --train_data datasets/ETDataset/ETT-small/ETTh1.csv \
  --split_ratio 0.2 \
  --output_path tmp/ts_test_1025 \
  --num_workers 4 \
  --label_col OT \
  --features_col "['HUFL','HULL','MUFL','MULL','LUFL','LULL']" \
  --params '{"data_format":"textfile","label_col":"OT","date_col":"date","features_col":["HUFL","HULL","MUFL","MULL","LUFL","LULL"],"freq":"h","seq_len":96,"pred_len":24,"moving_avg":25,"dropout":0.05,"train_epochs":3,"patience":2,"batch_size":32,"loss":"mse","sample_step":16}' \
