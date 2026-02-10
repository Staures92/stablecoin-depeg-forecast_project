for alpha in 0.4 1.0
do
for task in point quantile expectile
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --tau_pinball 0.025 \
    --model_name iTransformer \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "alpha_${alpha}_${task}" \
    --n_epochs 50 \
    --patience 10 \
    --verbose 1 \
    --check_lr 0 \
    --seq_len  168 \
    --pred_len 24 \
    --val_split 0.7 \
    --test_split 0.85 \
    --batch_size 256 \
    --test_batch_size 20 \
    --check_lr \
    --scaler revin \
    --affine 1 \
    --remote_logging \

done
done