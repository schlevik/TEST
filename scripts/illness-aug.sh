seq_len=104
model=GPT4TS

for pred_len in 24 36 48 60
do
for percent in 100
do

python main_LLM4TS.py \
    --root_path ./datasets_forecasting/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --name illness \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    --aug_only  1 \
    --output_dir ./experiments \
    --gpu -1 \
    --aug ili-clo-0_256_gen_100repeat.npy \
    --percent_aug -100

done
done

