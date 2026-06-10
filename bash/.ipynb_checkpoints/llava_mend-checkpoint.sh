#!/bin/bash
#SBATCH --job-name=llava_mend
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=mup_clip_lr_%j.out
#SBATCH --error=mup_clip_lr_%j.err                        


python multimodal_edit.py \
  --model llava \
  --method MEND \
  --train_json_path data/llava/train_dataset.json \
  --eval_json_path data/llava/val1_dataset.json \
  --mode train

