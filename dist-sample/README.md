

# Model Zoos
* DeepFM
* AutoInt
* DCN
* PNN

# Getting Started
### Train a model
You could optionally add extra command line parameters `--batch_size=${BATCH_SIZE}` and `--epochs=${EPOCHS}` to specify your preferred parameters. 
  

* Train with multiple GPUs on single machine
```shell script
bash tools/dist_train.sh 4 --data_path=t_megmind_train_data \
                           --feature_cols=c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --sparse_cols=d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --label_cols=label
```

* Train with multiple GPUs on multiple machines
```shell script
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.83.159.41 bash tools/dist_train.sh 4 \
                           --data_path=t_megmind_train_data \
                           --feature_cols=c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --sparse_cols=d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --label_cols=label
                           
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.83.159.41 bash tools/dist_train.sh 4 \
                           --data_path=t_megmind_train_data \
                           --feature_cols=c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --sparse_cols=d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                           --label_cols=label                      
```

* Train with a single GPU:
```shell script
bash tools/train.sh --data_path=t_megmind_train_data \
                    --feature_cols=c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                    --sparse_cols=d4,d18,d9,d12,d23,d1,d3,d17,d24,d13,d6,d15,d7,d26,d16,d20,d2,d10,d25,d14,d21,d5,d19,d22,d8,d11 \
                    --label_cols=label
```