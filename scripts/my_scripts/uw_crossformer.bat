@echo off
::set model_name=Autoformer
::set model_name=Informer
set model_name=Crossformer

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
  --pred_len 2 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 2 ^
  --c_out 5 ^
  --des "Exp" ^
  --d_model 512 ^
  --d_ff 2048 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 256 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1
::--debug
