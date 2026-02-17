for alpha in 1.0
do
for task in quantile expectile
do
for model in TSMixer
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name $model \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "${model}_alpha_${alpha}_${task}" \
    --n_epochs 50 \
    --patience 5 \
    --verbose 1 \
    --check_lr \
    --seq_len  168 \
    --pred_len 24 \
    --val_split 0.7 \
    --test_split 0.85 \
    --batch_size 512 \
    --test_batch_size 20 \
    --tau_pinball 0.025 \
    --dist_side both \
    --scaler revin \
    --affine 1 \
    --remote_logging \
    --n_cheb 8 \
    --dist_loss crps \
    --twcrps_side two_sided \
    --twcrps_smooth_h 2 \
    --u_grid_size 256 \

done
done
done
