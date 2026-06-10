#!/bin/bash

#########################
##### blip2 #####
#########################

# pip install transformers==4.37
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/MEND/blip2.yaml" \
#     --method "MEND" \
#     --train_json_path "data/blip2_new/train_dataset.json" \
#     --eval_json_path "data/blip2_new/val1_dataset.json" \
#     --mode "train"

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/blip2_new/train_dataset.json" \
#     --eval_json_path "data/blip2_new/val1_dataset.json" \
#     --mode "train"

pip install transformers==4.50

python multimodal_edit.py \
    --model "blip2" \
    --config_path "hparams/TRAINING/SERAC/blip2.yaml" \
    --method "SERAC" \
    --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
    --mode "train" \

# pip install transformers==4.37
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/SERAC/llava.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "train" \

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/FT/blip2.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "eval" \
    
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/FT/blip2_qformer.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "eval" \

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/MEND/blip2.yaml" \
#     --method "MEND" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "eval" \
#     --size 3

