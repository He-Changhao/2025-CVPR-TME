# dataset: CIRR or FashionIQ
# noise_ratio: the ratio of synthetic noise
# warmup_qformer: the first warmup phase for initialization.
# warmup_proj: the second warmup phase for stabilization.
# warmup_last: the third warmup phase for integration.
# partitioner: the partitioner to divide the train set into the noise part and clean part, default to GMM
# split_type: split the train set based on loss or similarity.
# save_training: whether to save the code
shuffle_seed=42
seed=42 
gpu=0
noise_ratio=0.0
python src/precompute_train.py \
    --exp_name "${your_exp_name}" \
    --shuffle_seed ${shuffle_seed} \
    --seed ${seed} \
    --dataset CIRR \
    --noise_ratio ${noise_ratio} \
    --nc_type "mix" \
    --batch_size 128 \
    --num_epochs 30 \
    --warmup_qformer 3 \
    --warmup_proj 2 \
    --warmup_last 1 \
    --partitioner "GMM" \
    --split_type "loss" \
    --threshold 0.5 \
    --lr "1e-5" \
    --lpm 1.0 \
    --lsa 1.0 \
    --lrd 0.2 \
    --save_training \
    --gpu ${gpu}