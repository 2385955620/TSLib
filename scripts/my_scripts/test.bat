@echo off
set CUDA_VISIBLE_DEVICES=3
set model_name=Crossformer

python -u run.py ^
  --task_name classification ^
  --is_training 1 ^
  --root_path ./dataset/EthanolConcentration/ ^
  --model_id EthanolConcentration ^
  --model %model_name% ^
  --data UEA ^
  --e_layers 3 ^
  --batch_size 16 ^
  --d_model 128 ^
  --d_ff 256 ^
  --top_k 3 ^
  --des "Exp" ^
  --itr 1 ^
  --learning_rate 0.001 ^
  --train_epochs 100 ^
  --patience 10

pause