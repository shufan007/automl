# export CUDA_VISIBLE_DEVICES=1

#cd ..

# ETT h1
python -u forecasting/run.py \
  --seed 1234 \
  --time_budget_s 300 \
  --num_samples 4 \
  --num_gpus_per_worker 1 \
  --resources_per_trial '{"cpu":1,"gpu":1}' \
  --search_space '{"model":["FEDformer","Autoformer"],"d_model":[64,512],"n_heads":[4,8],"d_ffn":[128,2048],"learning_rate":[0.00001,0.001]}' \
  --train_data datasets/ETDataset/ETT-small/ETTh1.csv \
  --split_ratio 0.2 \
  --output_path tmp/ts_test_0902 \
  --label_col OT \
  --date_col date \
  --features_col "['HUFL','HULL','MUFL','MULL','LUFL','LULL']" \
  --freq h \
  --seq_len 96 \
  --pred_len 24 \
  --dec_layers 1 \
  --moving_avg 25 \
  --dropout 0.05 \
  --num_workers 4 \
  --train_epochs 2 \
  --patience 2 \
  --batch_size 32 \
  --loss mse \
  --sample_step 16 \

