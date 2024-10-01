model_name=baseline

train_epochs=50
llama_layers=32
batch_size=24
learning_rate=0.001
d_model=8
d_ff=32

master_port=00097
num_process=8

comment='baseline-M4'

for season in Weekly Monthly Yearly Daily Quarterly Hourly; do 
  python run_m4_baseline.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns $season \
    --model_id "m4_$season" \
    --model $model_name \
    --data m4 \
    --features M \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --patch_len 1 \
    --stride 1 \
    --batch_size $batch_size \
    --des 'Exp' \
    --itr 1 \
    --learning_rate $learning_rate \
    --loss 'SMAPE' \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --gpt_layer 6 \
    --itr 3 \
    --is_gpt 1 \
    --output_dir ./experiments \
    --gpu -1
done