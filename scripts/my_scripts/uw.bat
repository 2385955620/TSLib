@echo off
set model_name=iTransformer

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 7.24_data.csv ^
  --model_id underwater_test ^
  --model %model_name% ^
  --data UnderWater ^
  --features M ^
  --seq_len 120 ^
  --label_len 60 ^
  --pred_len 1 ^
  --e_layers 3 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 10 ^
  --dec_in 10 ^
  --c_out 10 ^
  --des "Exp" ^
  --d_model 1024 ^
  --d_ff 4096^
  --itr 1 ^
  --freq S ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1 ^
  --lradj "cosine"
  
