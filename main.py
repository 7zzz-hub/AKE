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
    
    config_path = "/root/autodl-tmp/mllm_ke/akedit/config/llava.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Data related parameters
    parser.add_argument('--img_root', type=str, default=config.get('img_root', './datasets/images'), help='Root directory path for original images')
    parser.add_argument('--img_root_modified', type=str, default=config.get('img_root_modified', './datasets/modified_images'), help='Root directory path for modified images')
    parser.add_argument('--img_masks', type=str, default=config.get('img_masks', './datasets/images'), help='Root directory path for original images')    
    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path'), help='Dataset name')
    parser.add_argument('--image_direction_dataset_dir', type=str, default=config.get('image_direction_dataset_dir'), help='Dataset name')
    parser.add_argument('--image_size', type=str, default=config.get('image_size'))
    parser.add_argument('--patch_num', type=str, default=config.get('patch_num'))

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
    
    records_all = {}
    avg_results_all = {}


    tokenizer, image_processor, processor, \
        model, context_model, trg_model = classifier_helpers.load_classifier(args, layernum=args.edit_layers)

    AKEVLLMEditData = data_helpers.AKEVLLMEditData(image_processor, args)
    dataset = AKEVLLMEditData.get_dataset(args)

      
    val_loader = data_helpers.get_val_loader(args, dataset_dir=args.image_direction_dataset_dir)
        
    records, avg_result = edit.edit_model(
        args, processor, model, tokenizer, dataset,
        trg_model, context_model, val_loader, args.edit_layers)

    records_all[args.edit_layers] = records
    avg_results_all[args.edit_layers] = avg_result

    del model, context_model, trg_model, image_processor, processor
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    records_path = os.path.join(args.output_dir, f"records_{args.edit_layers}.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records_all, f, indent=2, ensure_ascii=False)

    avg_result_path = os.path.join(args.output_dir, f"avg_result_{args.edit_layers}.json")
    with open(avg_result_path, "w", encoding="utf-8") as f:
        json.dump(avg_results_all, f, indent=2, ensure_ascii=False)

        