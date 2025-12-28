@echo off
:: 设置显卡
set CUDA_VISIBLE_DEVICES=0

:: 设置超参数变量
set model_name=TimeMixer
set seq_len=120
set e_layers=2
set down_sampling_layers=3
set down_sampling_window=2
set learning_rate=0.01
set d_model=16
set d_ff=32
set train_epochs=10
set patience=10
set batch_size=16

:: 运行 Python 命令
:: 注意：Windows下换行符是 ^ 而不是 \
python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/underwater/ ^
  --data_path 12.27data.csv ^
  --model_id underwater_test ^
  --model %model_name% ^
  --data UnderWater ^
  --features M ^
  --seq_len %seq_len% ^
  --label_len 0 ^
  --pred_len 1 ^
  --e_layers %e_layers% ^
  --enc_in 8 ^
  --c_out 8 ^
  --des "Exp" ^
  --itr 1 ^
  --d_model %d_model% ^
  --d_ff %d_ff% ^
  --learning_rate %learning_rate% ^
  --train_epochs %train_epochs% ^
  --patience %patience% ^
  --batch_size 512 ^
  --down_sampling_layers %down_sampling_layers% ^
  --down_sampling_method avg ^
  --down_sampling_window %down_sampling_window% ^
  --num_workers 0 ^
  --lradj "cosine"

:: 运行结束后暂停，方便查看报错信息（如果一切正常可以去掉这一行）
pause