from helpers import context_helpers

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForVisualQuestionAnswering
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

    if config.model_name == 'qwenvl':
        from transformers import Qwen2_5_VLForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_path)
        processor = AutoProcessor.from_pretrained(config.model_path)
        image_processor = processor.image_processor

        model.requires_grad_(False)
        context_model = model.visual
        trg_model = model.visual.blocks[layernum].mlp.down_proj
        trg_model.requires_grad_(True)
    
    elif config.model_name == 'llava':
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=config.model_path,
            model_base=None,
            model_name="llava-v1.5-7b",
            torch_dtype=torch.float16)
        processor = image_processor
        
        model.model.requires_grad_(False)
        context_model = model.model.vision_tower
        trg_model = model.model.vision_tower.vision_tower.vision_model.encoder.layers[layernum].mlp.fc2
        trg_model.requires_grad_(True)

    elif config.model_name == 'blip2':
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        processor = AutoProcessor.from_pretrained(config.model_path)
        model = AutoModelForVisualQuestionAnswering.from_pretrained(config.model_path).cuda()
        image_processor = processor.image_processor

        model.requires_grad_(False)
        context_model = model.vision_model
        trg_model = model.vision_model.encoder.layers[layernum].mlp.fc2
        trg_model.requires_grad_(True)

    return tokenizer, image_processor, processor, model, context_model, trg_model

