

python export_onnx.py ^
  --checkpoint "checkpoints/long_term_forecast_underwater_test_iTransformer_UnderWater_ftM_sl120_ll60_pl1_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth" ^
  --task_name long_term_forecast ^
  --model iTransformer ^
  --seq_len 120 ^
  --label_len 60 ^
  --pred_len 1 ^
  --enc_in 8 ^
  --dec_in 2 ^
  --c_out 8 ^
  --d_model 512 ^
  --n_heads 8 ^
  --e_layers 3 ^
  --d_layers 1 ^
  --d_ff 2048 ^
  --factor 3 ^
  --embed timeF ^
  --freq h ^
  --dynamic_batch ^
  --device cpu