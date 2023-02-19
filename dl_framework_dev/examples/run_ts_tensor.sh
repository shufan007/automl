
#export CUDA_VISIBLE_DEVICES=1

#cd ..

# ETT h1
python -u forecasting/run.py \
  --seed 12345 \
  --time_budget_s 300 \
  --num_samples 4 \
  --num_gpus_per_worker 1 \
  --search_space '{"model":["FEDformer","Autoformer"],"d_model":[32,64],"n_heads":[2,4],"d_ffn":[32,64],"learning_rate":[0.00001,0.001]}' \
  --data_format tensor \
  --train_data datasets/battery_cost/seq_d_list.pt \
  --split_ratio 0.2 \
  --output_path fanshuangxi/tmp/ts_test \
  --label_col y \
  --seq_len 49 \
  --pred_len 1 \
  --moving_avg 25 \
  --train_epochs 2 \
  --patience 2 \
  --batch_size 32 \
  --loss mse \
  --sample_step 16 \
