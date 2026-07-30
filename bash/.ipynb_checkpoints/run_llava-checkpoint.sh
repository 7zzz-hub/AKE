#!/bin/bash

#########################
##### LLaVA #####
#########################

pip install transformers==4.37

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava_mmproj.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --size 10

python multimodal_edit.py \
    --model "llava" \
    --config_path "hparams/FT/llava.yaml" \
    --method "FT-L" \
    --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \
    --size 10

## mend+clevr
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --size 10


### serac+clevr
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/SERAC/llava.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/SERAC/llava.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val2_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
    
### FTL+clevr
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \

# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/clevr_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/llava/val2_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \

### ftl+supple
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/FT/llava.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "eval" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 \

# ### mend+supple
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/MEND/llava.yaml" \
#     --method "MEND" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 


# ### serac+supple
# python multimodal_edit.py \
#     --model "llava" \
#     --config_path "hparams/TRAINING/SERAC/llava.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/supple_dataset/llava/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/llava/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 \
#     --checkpoint "results/models/SERAC_MULTI/llava_serac_supple-step_12000.pt"