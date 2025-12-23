@echo off


:: 设置变量 (对应 model_name=TimeXer)
set model_name=TimeXer

:: 执行 Python 命令
:: 注意：Linux 的换行符 \ 变成了 Windows 的 ^
:: 变量引用 $model_name 变成了 %model_name%
:: 单引号 'exp' 建议改为双引号 "exp" 以兼容 Windows CMD

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_96_96 ^
  --model %model_name% ^
  --data ETTh1 ^
  --features M ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --d_model 256 ^
  --batch_size 4 ^
  --des "exp" ^
  --itr 1

pause