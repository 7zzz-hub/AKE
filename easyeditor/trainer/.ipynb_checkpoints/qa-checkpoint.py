from PIL import Image
import torch

def answer_single_question(model_class, vis_processor, edited_model, batch):

    if model_class == "LLaVA":
        image = [vis_processor(Image.open(img_path), return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16) for img_path in batch["image"]]
    else:
        image = [vis_processor(Image.open(img_path).convert("RGB")).to(dtype=torch.float16) for img_path in batch["image"]]

    batch["image"] = torch.stack(image, dim=0)
    post_edit_outputs = edited_model(batch)
    if not isinstance(post_edit_outputs, torch.Tensor):
        post_edit_logits = post_edit_outputs.logits
    else:  
        post_edit_logits = post_edit_outputs
    post_batch_labels = batch["labels"]

    return post_edit_logits, post_batch_labels
    

# def answer_multi_question(model_class, vis_processor, edited_model, batch):

#     tmp = []
#     for _ in range(len(batch)):
#         for idx in range(len(batch['labels'][_])):
#             tmp.append({
#                 "image": [batch["image"][_][idx]],
#                 "text_input": [batch["text_input"][_][idx]],
#                 "prompts_len": [batch["prompts_len"][_][idx]],
#                 "labels": batch["labels"][_][idx]
#             })
    
#     post_edit_logits=[]; post_batch_labels=[]
#     for b in tmp:
#         _logit, _label = answer_single_question(model_class, vis_processor, edited_model, b)
#         post_edit_logits.append(_logit)
#         post_batch_labels.append(_label)
        
#     return post_edit_logits, post_batch_labels

def answer_multi_question(model_class, vis_processor, edited_model, batch):

    post_edit_logits=[]
    for idx in range(len(batch['labels'])):
        tmp = {}
        for k in batch:
            if k=="image":
                images = [Image.open(img_path).convert("RGB") for img_path in batch[k][idx]]
                if model_class == "LLaVA":
                    tmp[k] = torch.stack([vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                else:
                    tmp[k] = torch.stack([vis_processor(image).to(dtype=torch.float16) for image in images], dim=0)
            else:
                tmp[k] = batch[k][idx]
        post_base_outputs = edited_model(tmp)
        if not isinstance(post_base_outputs, torch.Tensor):
            post_edit_logits.append(post_base_outputs.logits)
        else:
            post_edit_logits.append(post_base_outputs)
    post_batch_labels = batch['labels']
    
    return post_edit_logits, post_batch_labels
   

def compute_score(config, model, post_edit_logits, post_batch_labels):
    post_edit_dict = []
    
    for _post_edit_logits, _post_batch_labels in zip(post_edit_logits, post_batch_labels):
        if _post_edit_logits.shape[1] > _post_batch_labels.shape[1]:
            post_edit_dict.append(model.edit_loss_fn(config, _post_edit_logits, _post_batch_labels))
        else:
            post_edit_dict.append(model.edit_loss_fn(config, _post_edit_logits, _post_batch_labels[:, -_post_edit_logits.shape[1]-1:]))

    return post_edit_dict