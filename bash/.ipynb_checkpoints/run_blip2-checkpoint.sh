#!/bin/bash

#########################
##### Blip2 #####
#########################

pip install transformers==4.50

### ftl+clevr
python multimodal_edit.py \
    --model "blip2" \
    --config_path "hparams/FT/blip2.yaml" \
    --method "FT-L" \
    --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/blip2/val2_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \

python multimodal_edit.py \
    --model "blip2" \
    --config_path "hparams/FT/blip2_qformer.yaml" \
    --method "FT-V" \
    --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/FT/blip2.yaml" \
#     --method "FT-L" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "eval" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 


### ftv+clevr
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/FT/blip2_qformer.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
    
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/FT/blip2_qformer.yaml" \
#     --method "FT-V" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val2_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \

### mend+clevr
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/MEND/blip2.yaml" \
#     --method "MEND" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
    # --checkpoint "results/models/MEND/blip2_mend_clevr-step_3000.pt"

python multimodal_edit.py \
    --model "blip2" \
    --config_path "hparams/MEND/blip2.yaml" \
    --method "MEND" \
    --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/blip2/val2_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \

python multimodal_edit.py \
    --model "blip2" \
    --config_path "hparams/MEND/blip2.yaml" \
    --method "MEND" \
    --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
    --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
    --mode "eval" \
    --dataset_type "AttributeDataset" \
    --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
    --device 0 \

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --size 100

### serac+clevr
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --size 10
    
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "train" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \
#     --checkpoint "results/models/SERAC_MULTI/blip2_260624_202940-step_5000.pt"

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/clevr_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/clevr_dataset/blip2/val1_dataset.json" \
#     --mode "eval" \
#     --dataset_type "AttributeDataset" \
#     --image "data/clevr_dataset/CLEVR_CoGenT_v1.0/images" \
#     --device 0 \    

# ### mend+supple
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/MEND/blip2.yaml" \
#     --method "MEND" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 


# ### serac+supple
# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/TRAINING/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "train" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 

# python multimodal_edit.py \
#     --model "blip2" \
#     --config_path "hparams/SERAC/blip2.yaml" \
#     --method "SERAC" \
#     --train_json_path "data/supple_dataset/blip2/train_dataset.json" \
#     --eval_json_path "data/supple_dataset/blip2/eval_dataset.json" \
#     --mode "eval" \
#     --dataset_type "SuppleDataset" \
#     --image "data/supple_dataset/" \
#     --t_loc_image "data/supple_dataset/black_image.png" \
#     --device 0 \
#     --size 10