################## Qwen3-VL ##################

python vead_train.py \
-mn qwenvl \
-bs 1 \
-dvc cuda:0 \
-dp data/clevr_dataset/qwenvl/train_dataset.json \
-img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
-dataset_type AttributeDataset \

python vead_train.py \
-mn qwenvl \
-bs 1 \
-dvc cuda:0 \
-dp data/supple_dataset/qwenvl/train_dataset.json \
-img_root data/supple_dataset/ \
--t_loc_image data/supple_dataset/black_image.png \
-dataset_type SuppleDataset \


# ################## BLIP2 ##################

python vead_train.py \
-mn blip2 \
-bs 1 \
-dvc cuda:0 \
-dp data/clevr_dataset/blip2/train_dataset.json \
-img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
-dataset_type AttributeDataset \

python vead_train.py \
-mn blip2 \
-bs 1 \
-dvc cuda:0 \
-dp data/supple_dataset/blip2/train_dataset.json \
-img_root data/supple_dataset/ \
--t_loc_image data/supple_dataset/black_image.png \
-dataset_type SuppleDataset \


# ################## LLAVA ##################

python vead_train.py \
-mn llava \
-bs 1 \
-dvc cuda:0 \
-dp data/clevr_dataset/llava/train_dataset.json \
-img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
-dataset_type AttributeDataset \

python vead_train.py \
-mn llava \
-bs 1 \
-dvc cuda:0 \
-dp data/supple_dataset/llava/train_dataset.json \
-img_root data/supple_dataset/ \
--t_loc_image data/supple_dataset/black_image.png \
-dataset_type SuppleDataset \
