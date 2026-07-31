################## QwenVL ##################

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

# python vead_test.py \
# -mn qwenvl \
# -dvc cuda:0 \
# -ckpt /root/autodl-tmp/VisEdit/records/vead/qwenvl/2026.07.02-17.03.48/checkpoints/epoch-1-i-900-ema_loss-3.9654 \
# -dp data/clevr_dataset/qwenvl/val2_dataset.json \
# -img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -loc_img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -dataset_type AttributeDataset \
# -dsn 10


################## BLIP2 ##################

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

# python vead_test.py \
# -mn blip2 \
# -dvc cuda:0 \
# -ckpt /root/autodl-tmp/VisEdit/records/vead/blip2/2026.07.02-11.33.03/checkpoints/epoch-1-i-1000-ema_loss-0.4024 \
# -dp data/clevr_dataset/blip2/val2_dataset.json \
# -img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -loc_img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -dataset_type AttributeDataset \
# -dsn 10

################## LLAVA ##################

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

# python vead_test.py \
# -mn llava \
# -dvc cuda:0 \
# -ckpt /root/autodl-tmp/VisEdit/records/vead/llava-v1.5-7b/2026.05.26-09.31.45/checkpoints/epoch-3-i-7500-ema_loss-0.4901 \
# -dp data/clevr_dataset/llava/val2_dataset.json \
# -img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -loc_img_root data/clevr_dataset/CLEVR_CoGenT_v1.0/images \
# -dataset_type AttributeDataset \

