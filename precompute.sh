nohup python src/precompute.py \
    --dataset 'CIRR' \
    --split 'val' \
    --batch_size '512' \
    --gpu 2 \
    --log_file_name precompute_test \
    > run.log 2>&1 &