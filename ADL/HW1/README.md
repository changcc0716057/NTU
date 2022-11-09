# Step for training models

## Environment
```shell
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent Classification
### Training
```shell
python train_intent.py [-h] [--data_dir DATA_DIR] [--cache_dir CACHE_DIR] [--ckpt_dir CKPT_DIR] [--max_len MAX_LEN] [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL] [--lr LR] [--batch_size BATCH_SIZE] [--device DEVICE] [--num_epoch NUM_EPOCH]
```
### Testing
```shell
python test_intent.py [-h] --test_file TEST_FILE [--cache_dir CACHE_DIR] --ckpt_path CKPT_PATH [--pred_file PRED_FILE] [--max_len MAX_LEN][--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL] [--batch_size BATCH_SIZE] [--device DEVICE]
```

## Slot Tagging
### Training
```shell
python train_slot.py [-h] [--data_dir DATA_DIR] [--cache_dir CACHE_DIR] [--ckpt_dir CKPT_DIR] [--max_len MAX_LEN] [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL] [--lr LR] [--batch_size BATCH_SIZE] [--device DEVICE] [--num_epoch NUM_EPOCH]
```
### Testing
```shell
python test_slot.py [-h] [--test_file TEST_FILE] [--cache_dir CACHE_DIR] [--ckpt_path CKPT_PATH] [--pred_file PRED_FILE] [--max_len MAX_LEN] [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--dropout DROPOUT] [--bidirectional BIDIRECTIONAL] [--batch_size BATCH_SIZE] [--device DEVICE]
```