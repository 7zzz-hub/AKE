from helpers import context_helpers
from helpers.qwen_helpers import get_qwen_mlp_output, get_qwen_visual, is_qwen_vl

import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor, BlipImageProcessor
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def eval_accuracy(model, loader, alt=False, normalize=None):
    labels, preds = [], []
    with torch.no_grad():
        for _, (im, targ) in enumerate(loader):
            if normalize:
                im = normalize(im.cuda())
            if alt:
                op, _ = model(im.cuda())
            else:
                op = model(im.cuda()) #model(normalizer(im))
            preds.append(op.argmax(dim=1).cpu())
            labels.append(targ)
    return torch.cat(preds), torch.cat(labels)

    
def load_classifier(config, layernum):

    if is_qwen_vl(config.model_name):
        if not os.path.isdir(config.model_path):
            raise FileNotFoundError(
                f"Qwen-VL model directory does not exist: {config.model_path}")
        import transformers
        if not hasattr(transformers, "AutoModelForImageTextToText"):
            raise RuntimeError(
                "Installed transformers does not provide the Qwen-VL auto model")
        model_class = transformers.AutoModelForImageTextToText
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
        model = model_class.from_pretrained(
            config.model_path, dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(config.model_path)
        image_processor = processor.image_processor

        model.requires_grad_(False)
        context_model = get_qwen_visual(model)
        if not 0 <= layernum < len(context_model.blocks):
            raise ValueError(
                f"edit_layers={layernum} is outside the Qwen-VL vision tower "
                f"with {len(context_model.blocks)} blocks")
        trg_model = get_qwen_mlp_output(context_model.blocks[layernum])
        trg_model.requires_grad_(True)

    elif config.model_name == 'llava':
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=config.model_path,
            model_base=None,
            model_name="llava-v1.5-7b",
            torch_dtype=torch.float16)
        processor = image_processor
        
        model.requires_grad_(False)
        context_model = model.model.vision_tower
        trg_model = model.model.vision_tower.vision_tower.vision_model.encoder.layers[layernum].mlp.fc2
        trg_model.requires_grad_(True)

    elif config.model_name == 'blip2':
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
        image_processor = BlipImageProcessor.from_pretrained(config.model_path)
        processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
        model = Blip2ForConditionalGeneration.from_pretrained(
            config.model_path, torch_dtype=torch.float16).cuda()
        processor.num_query_tokens = model.config.num_query_tokens
        image_processor = processor.image_processor

        model.requires_grad_(False)
        context_model = model.vision_model
        trg_model = model.vision_model.encoder.layers[layernum].mlp.fc2
        trg_model.requires_grad_(True)

    return tokenizer, image_processor, processor, model, context_model, trg_model

