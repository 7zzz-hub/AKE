import gc
import json
import argparse, os, warnings

import torch
from tqdm import trange
from transformers import AutoProcessor, AutoTokenizer
import yaml
warnings.filterwarnings("ignore")

from helpers import classifier_helpers, data_helpers, rewrite_helpers
import edit


def parse_args():
    parser = argparse.ArgumentParser(description='AKE VLLM Edit Training')
    
    config_path = "/root/autodl-tmp/vision_edit/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Data related parameters
    parser.add_argument('--img_root', type=str, default=config.get('img_root', './datasets/images'), help='Root directory path for original images')
    parser.add_argument('--img_root_modified', type=str, default=config.get('img_root_modified', './datasets/modified_images'), help='Root directory path for modified images')
    parser.add_argument('--img_masks', type=str, default=config.get('img_masks', './datasets/images'), help='Root directory path for original images')    
    
    parser.add_argument('--train_dataset_path', type=str, default=config.get('train_dataset_path'), help='Dataset name')
    parser.add_argument('--eval_dataset_path', type=str, default=config.get('eval_dataset_path'), help='Dataset name')
    parser.add_argument('--image_direction_dataset_dir', type=str, default=config.get('image_direction_dataset_dir'), help='Dataset name')


    # Editing method parameters
    parser.add_argument('--edit_layers', type=int, default=config.get('edit_layers', None), help='Layers to be edited')

    # Model related parameters
    parser.add_argument('--model_name', type=str, default=config.get('model_name'), help='Model name')
    parser.add_argument('--model_path', type=str, default=config.get('model_path'), help='Model name')
    parser.add_argument('--vision_model_path', type=str, default=config.get('vision_model_path'), help='Model name')

    # Training related parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    # Path parameters
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir'), help='Output directory')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    val_loader = data_helpers.get_val_loader(dataset_dir=args.image_direction_dataset_dir)
    
    records_all = {}
    avg_results_all = {}

    for layernum in trange(8, 9):
        
        print(f"\n=== 🧠 正在处理层: {layernum} ===")

        tokenizer, image_processor, processor, model, context_model, trg_model = classifier_helpers.load_classifier(
            model_path=args.model_path,
            layernum=layernum
        )

        AKEVLLMEditData = data_helpers.AKEVLLMEditData(image_processor, args)
        train_data = AKEVLLMEditData.get_dataset(args)

        records, avg_result = edit.edit_model(
            processor, model, tokenizer, train_data,
            trg_model, context_model, val_loader, layernum)

        records_all[layernum] = records
        avg_results_all[layernum] = avg_result

        del model, context_model, trg_model, image_processor, processor
        torch.cuda.empty_cache()
        gc.collect()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    records_path = os.path.join(args.output_dir, "records_all_layers.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records_all, f, indent=2, ensure_ascii=False)

    avg_result_path = os.path.join(args.output_dir, "avg_result_all_layers.json")
    with open(avg_result_path, "w", encoding="utf-8") as f:
        json.dump(avg_results_all, f, indent=2, ensure_ascii=False)

        