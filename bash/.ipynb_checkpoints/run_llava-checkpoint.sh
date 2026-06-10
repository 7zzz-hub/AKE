#!/bin/bash

#########################
##### LLaVA #####
#########################

# pip install transformers==4.37
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/llava_new/train_dataset.json" \
#     --eval_json_path "data/llava_new/val1_dataset.json" \
#     --mode "train"

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/llava_new/train_dataset.json" \
#     --eval_json_path "data/llava_new/val1_dataset.json" \
#     --mode "eval"

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/llava_new/train_dataset.json" \
#     --eval_json_path "data/llava_new/val2_dataset.json" \
#     --mode "eval"




# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "eval"

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava_mmproj.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "eval"

# pip install transformers==4.37
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava_mmproj.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/llava_new/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava_new/val1_dataset.json" \
#     --mode "eval" \
#     --size 25

python multimodal_edit.py \
    --model "llava" \
    --config_path "hparams/MEND/llava.yaml" \
    --method "MEND" \
    --train_json_path "data/supple_dataset/llava/train_dataset.json" \
    --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
    --mode "eval" \

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava_mmproj.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "eval"