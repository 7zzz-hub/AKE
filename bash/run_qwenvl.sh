#!/bin/bash

#########################
##### Qwen-vl #####
#########################

# pip install transformers==4.52

# OMP_NUM_THREADS=8 python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/FT/qwenvl_vit.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --size 10

OMP_NUM_THREADS=8 python multimodal_edit.py \
    --model "qwen-vl" \
    --config_path "hparams/FT/qwenvl.yaml" \
    --method "FT-L" \
    --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \
    --size 10


### mend+clevr
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/TRAINING/MEND/qwenvl.yaml" \
#     --method "MEND" \
#     --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 

# ### serac+clevr
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/TRAINING/SERAC/qwenvl.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 


# ### mend+supple
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/TRAINING/MEND/qwenvl.yaml" \
#     --method "MEND" \
#     --train_json_path "data/supple_dataset/qwen/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/qwen/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 


# ### serac+supple
# python multimodal_edit.py \
#     --model "qwen-vl" \
#     --config_path "hparams/TRAINING/SERAC/qwenvl.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/supple_dataset/qwen/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/qwen/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 