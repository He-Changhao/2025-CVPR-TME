# mode: validate or test; 'test' only for the CIRR dataset
# dataset: CIRR or FashionIQ
# gpu: the index of gpu to be used, default to 0
# name: the name of folder; the folder will be output to the weight directory
python precompute_test.py \
    --dataset "CIRR" \
    --mode "validate" \
    --model-path "${your_model_path}" \
    --gpu "$gpu" \
    --name "${save_folder_name}"