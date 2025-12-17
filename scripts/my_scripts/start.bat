:: 1. 设置变量
set model_name=iTransformer

:: 2. 运行命令（多行版本，注意 ^ 后面不要有空格）
python -u run.py ^
  --task_name short_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/m4 ^
  --seasonal_patterns "Hourly" ^
  --model_id m4_Hourly ^
  --model %model_name% ^
  --data m4 ^
  --features M ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 1 ^
  --dec_in 1 ^
  --c_out 1 ^
  --batch_size 16 ^
  --d_model 512 ^
  --des "Exp" ^
  --itr 1 ^
  --learning_rate 0.001 ^
  --loss "SMAPE"