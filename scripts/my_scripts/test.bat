@echo off


:: 设置变量 (对应 model_name=TimeXer)
::set model_name=TimeXer
set model_name=TimeMixer
:: 执行 Python 命令
:: 注意：Linux 的换行符 \ 变成了 Windows 的 ^
:: 变量引用 $model_name 变成了 %model_name%
:: 单引号 'exp' 建议改为双引号 "exp" 以兼容 Windows CMD

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
  --e_layers 1 ^
  --factor 3 ^
  --enc_in 8 ^
  --dec_in 7 ^
  --c_out 5 ^
  --d_model 256 ^
  --batch_size 512 ^
  --des "exp" ^
  --itr 1 ^
  --train_epochs 20 ^
  --patience 10 ^
  --num_workers 0 ^
  --lradj "cosine"

pause