#!/bin/bash

#########################
##### Qwen-vl #####
#########################

pip install transformers==4.52


### mend+clevr
python multimodal_edit.py \
    --model "qwen-vl" \
    --config_path "hparams/TRAINING/MEND/qwenvl.yaml" \
    --method "MEND" \
    --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
    --mode "train" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 

### serac+clevr
python multimodal_edit.py \
    --model "qwen-vl" \
    --config_path "hparams/TRAINING/SERAC/qwenvl.yaml" \
    --method "SERAC" \
    --train_json_path "data/clevr_dataset/qwenvl/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/qwenvl/val1_dataset.json" \
    --mode "train" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 


### mend+supple
python multimodal_edit.py \
    --model "qwen-vl" \
    --config_path "hparams/TRAINING/MEND/qwenvl.yaml" \
    --method "MEND" \
    --train_json_path "data/supple_dataset/qwen/train_dataset.json" \
    --eval_json_path "data/supple_dataset/qwen/eval_dataset.json" \
    --mode "train" \
    --dataset_type "SuppleDataset" \
    --image "data/supple_dataset/" \
    --t_loc_image "data/supple_dataset/black_image.png" \
    --device 0 


### serac+supple
python multimodal_edit.py \
    --model "qwen-vl" \
    --config_path "hparams/TRAINING/SERAC/qwenvl.yaml" \
    --method "SERAC" \
    --train_json_path "data/supple_dataset/qwen/train_dataset.json" \
    --eval_json_path "data/supple_dataset/qwen/eval_dataset.json" \
    --mode "train" \
    --dataset_type "SuppleDataset" \
    --image "data/supple_dataset/" \
    --t_loc_image "data/supple_dataset/black_image.png" \
    --device 0 