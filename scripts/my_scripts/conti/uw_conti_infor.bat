@echo off
::set model_name=Autoformer
set model_name=Informer

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 0 ^
  --root_path ./dataset/underwater/ ^
  --data_path extracted_rows.csv ^
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
  --enc_in 8 ^
  --dec_in 2 ^
  --c_out 5 ^
  --des "Exp" ^
  --d_model 1024 ^
  --d_ff 4096 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 0 ^
  --lradj "cosine"