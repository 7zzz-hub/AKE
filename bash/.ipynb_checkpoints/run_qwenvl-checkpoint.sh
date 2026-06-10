#!/bin/bash
#########################
##### QWEN-VL #####
#########################
# pip install transformers==4.50

# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/FT/qwenvl.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val1_dataset.json" \
#     --mode "eval"

# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/FT/qwenvl.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val2_dataset.json" \
#     --mode "eval"

# pip install transformers==4.50
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/FT/qwenvl_vit.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val1_dataset.json" \
#     --mode "eval"

# pip install transformers==4.50
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/FT/qwenvl_vit.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val2_dataset.json" \
#     --mode "eval"

pip install transformers==4.37
python multimodal_edit.py \
    --model "llava" \
    --config_path "hparams/FT/llava.yaml" \
    --method "FT-L" \
    --train_json_path "data/llava/train_dataset.json" \
    --eval_json_path "data/llava/val2_dataset.json" \
    --mode "eval"
    
pip install transformers==4.37
python multimodal_edit.py \
    --model "llava" \
    --config_path "hparams/FT/llava_mmproj.yaml" \
    --method "FT-V" \
    --train_json_path "data/llava/train_dataset.json" \
    --eval_json_path "data/llava/val2_dataset.json" \
    --mode "eval"
    
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/TRAINING/MEND/qwenvl.yaml" \
#     --method "MEND" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val1_dataset.json" \
#     --mode "train"

# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/MEND/qwenvl.yaml" \
#     --method "MEND" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val1_dataset.json" \
#     --mode "eval"

# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/MEND/qwenvl.yaml" \
#     --method "MEND" \
#     --train_json_path "data/qwen/train_dataset.json" \
#     --eval_json_path "data/qwen/val2_dataset.json" \
#     --mode "eval"