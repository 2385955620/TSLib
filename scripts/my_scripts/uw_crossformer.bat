@echo off
::set model_name=Autoformer
::set model_name=Informer
set model_name=Crossformer

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
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
  --d_model 64 ^
  --d_ff 256 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1 ^
  --lradj "cosine"

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
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
  --d_model 128 ^
  --d_ff 512 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1 ^
  --lradj "cosine"


python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
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
  --d_model 256 ^
  --d_ff 1024 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1 ^
  --lradj "cosine"

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
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
  --d_model 512 ^
  --d_ff 2048 ^
  --itr 1 ^
  --freq s ^
  --data_stride 20 ^
  --batch_size 512 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 1 ^
  --lradj "cosine"

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
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
  --num_workers 1 ^
  --lradj "cosine"
::--debug
