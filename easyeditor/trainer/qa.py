from datetime import datetime
import os
from PIL import Image
import torch
import copy
from qwen_vl_utils import process_vision_info


def build_qwenvl_message(image, question):
    return {
        "role": "user",
        "content": [
            # {"type": "image", "image": image, "resized_height": 336, "resized_width": 336},
            {"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1280*28*28,},
            {
                "type": "text",
                "text": question + " Please answer in one word."
            }
        ]
    }

def prepare_inputs(config, vis_processor, batch):

    sample = copy.deepcopy(batch)

    if config.model_class == "LLaVA":
        images = [
            vis_processor(Image.open(img_path), return_tensors="pt")["pixel_values"].squeeze(0)
            for img_path in sample["image"]
        ]
        sample["image"] = torch.stack(images, dim=0).to(dtype=torch.float16)
    elif config.model_class == "qwen-vl":
        messages = [
            build_qwenvl_message(img, prompt)
            for img, prompt in zip(sample["image"], sample["prompts"])
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        processor_outputs = vis_processor(
            text=sample["text_input"],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        sample["inputs"] = processor_outputs
        sample["text_input"] = batch["text_input"]
    else:
        images = [
            vis_processor(Image.open(img_path).convert("RGB"))
            for img_path in sample["image"]
        ]
        sample["image"] = torch.stack(images, dim=0).to(dtype=torch.float16)

    return sample


def forward_model(model, batch, save_states=False):

    outputs = model(batch)
    if isinstance(outputs, torch.Tensor):
        logits = outputs
        attention_mask = torch.ones(logits.shape[:2], device=logits.device)
    else:
        logits = outputs.logits
        if hasattr(outputs, "attention_mask"):
            attention_mask = outputs.attention_mask
        else:
            attention_mask = torch.ones(logits.shape[:2], device=logits.device)

    ## hidden states
    # if save_states:     
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_dir = "hidden_states"
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(
    #         save_dir,
    #         f"/root/autodl-tmp/mllm_ke/AKE-main/hidden_states/hidden_25/FT-L/hidden_states_{timestamp}.pt"
    #     )
    #     hidden_states = [
    #         h[:,-batch['labels'].shape[1]:,:].detach().float().cpu()
    #         for h in outputs.hidden_states
    #     ]
    #     torch.save(hidden_states, save_path)

    # if save_states:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #     save_dir = "/root/autodl-tmp/mllm_ke/AKE-main/hidden_states/attention_akedit/FT-L"
    #     os.makedirs(save_dir, exist_ok=True)
    
    #     save_path = os.path.join(save_dir, f"attentions_{timestamp}.pt")
    
    #     # outputs.attentions:
    #     attentions = [
    #         attn.detach().float().cpu()
    #         for attn in outputs.output_attentions
    #     ]
    
    #     save_obj = {
    #         "attentions": attentions,
    #         "input_ids": batch.get("input_ids", None).detach().cpu()
    #             if batch.get("input_ids", None) is not None else None,
    #         "labels": batch.get("labels", None).detach().cpu()
    #             if batch.get("labels", None) is not None else None,
    #         "attention_mask": batch.get("attention_mask", None).detach().cpu()
    #             if batch.get("attention_mask", None) is not None else None,
    #     }
    
    #     torch.save(save_obj, save_path)
    
    #     print(f"Saved full attentions to {save_path}")

    return logits, attention_mask


def answer_single_question(config, vis_processor, model, batch, save_states=False):

    all_logits=[]; all_labels=[]
    for b in batch:
        sample = prepare_inputs(config, vis_processor, b)
        logits, _ = forward_model(model, sample, save_states=save_states)
        all_logits.append(logits)
        all_labels.append(sample["labels"])
    
    post_edit_logits = torch.cat(all_logits, dim=0)
    post_batch_labels = torch.cat(all_labels, dim=0)

    return post_edit_logits, post_batch_labels


def compute_single_score(config, model, post_edit_logits, post_batch_labels, batch, record):

    if record == {}:
        record.update({key: [] for key in ['image', 'prompt', 'target',  'post_edit']})

    if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
        post_edit_dict = model.edit_loss_fn(config, post_edit_logits, post_batch_labels)
    else:
        post_edit_dict = model.edit_loss_fn(config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])

    for b in batch:
        record['image'].extend(b['image'])
        record['prompt'].extend(b['text_input'])
    
    record['target'].extend(post_edit_dict['targ_token'])
    record['post_edit'].extend(post_edit_dict['pred_token'])

    return post_edit_dict

def process_image(config, vis_processor, batch):
    # image
    if config.model_class == "LLaVA":
        batch["image"] = torch.stack([
            vis_processor(Image.open(p), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            for p in batch["image"]
        ], dim=0)
    elif config.model_class == "qwen-vl":
        batch["image"], _ = process_vision_info([build_qwenvl_message(image, p) for image, p in zip(batch['image'], batch['prompts'])])
    else:
        batch["image"] = torch.stack([
            vis_processor(Image.open(p).convert("RGB")).to(dtype=torch.float16)
            for p in batch["image"]
        ], dim=0)  
        
    return batch


def edit_loc_data(config, vis_processor, model, kl_loc_loss, base_logits, loc_data):
    
    post_base_logits=[]; kl_mask=[]
    for loc in loc_data:
        
        loc = prepare_inputs(config, vis_processor, loc)

        post_logits, attention_mask = forward_model(model, loc)

        post_base_logits.append(post_logits)
        kl_mask.append(attention_mask)

    post_base_logits = torch.cat(post_base_logits, dim=0)
    kl_mask = torch.cat(kl_mask, dim=0)

    l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

    return l_loc, post_base_logits