@echo off
set model_name=iTransformer

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 7.24_data.csv ^
  --model_id underwater_96_48 ^
  --model %model_name% ^
  --data UnderWater ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 48 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 10 ^
  --dec_in 10 ^
  --c_out 10 ^
  --des "Exp" ^
  --d_model 128 ^
  --d_ff 128 ^
  --itr 1 ^
  --freq S ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --debug