for alpha in 0.4 1.0
do
for method in earlywarning
do

python main_lightning.py \
    --alpha $alpha \
    --model_name iTransformer \
    --method $method \
    --target_threshold 15 \
    --target_window 24 \
    --depeg_side both \
    --experiment_name stablecoin-depeg \
    --run_name "alpha_${alpha}_${method}" \
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
    --compute_shap 1 \
    --shap_background_size 64 \
    --shap_test_samples 256 \
    --scaler revin \
    --affine 1 \
    --remote_logging \

done
done