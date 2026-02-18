for alpha in 0.4 1.0 1.5
do
for loss in bce focal
do
for model in CNN
do

python main_lightning.py \
    --alpha $alpha \
    --model_name $model \
    --method earlywarning \
    --target_threshold 15 \
    --target_window 24 \
    --depeg_side both \
    --experiment_name stablecoin-earlywarning \
    --run_name "alpha_${alpha}_model_${model}" \
    --n_epochs 50 \
    --patience 10 \
    --verbose 1 \
    --check_lr 0 \
    --seq_len  168 \
    --pred_len 1 \
    --val_split 0.6 \
    --test_split 0.8 \
    --batch_size 256 \
    --test_batch_size 20 \
    --check_lr  \
    --compute_shap 0 \
    --shap_background_size 64 \
    --shap_test_samples 256 \
    --class_loss $loss \
    --scaler revin \
    --remote_logging \

done
done
done