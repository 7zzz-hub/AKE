from argparse import Namespace
import gc
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from helpers import rewrite_helpers, context_helpers
from helpers.qwen_helpers import get_qwen_visual, is_qwen_vl
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def _qwen_inputs(processor, image_path, prompt, target=None):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=target is None)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt")


def _one_word_prompt(prompt):
    suffix = "Please answer in one word."
    prompt = prompt.strip()
    return prompt if prompt.endswith(suffix) else f"{prompt} {suffix}"


def teacher_forcing_token_loss(config, model, tokenizer, processor, train_data):
    """Supervise answer tokens only for each supported VLM."""
    if config.model_name == "llava":
        prompt = (f"USER: {DEFAULT_IMAGE_TOKEN}\n {train_data['prompt']} "
                  "Please answer in one word. ASSISTANT:")
        prompt_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        answer_ids = tokenizer(
            " " + train_data["target"], add_special_tokens=False,
            return_tensors="pt").input_ids[0]
        if tokenizer.eos_token_id is not None:
            answer_ids = torch.cat([answer_ids, torch.tensor(
                [tokenizer.eos_token_id], dtype=answer_ids.dtype)])
        input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(model.device)
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_ids.numel():] = input_ids[:, prompt_ids.numel():]
        dtype = next(model.model.mm_projector.parameters()).dtype
        images = train_data["imgs"][:1].to(model.device, dtype=dtype)
        return model(input_ids=input_ids, labels=labels, images=images,
                     use_cache=False, return_dict=True).loss

    if config.model_name == "blip2":
        image = Image.open(train_data["image"]).convert("RGB")
        prompt = (f"Question: {train_data['prompt']} Please answer in one word. "
                  "Short answer:")
        encoded = processor(images=image, text=prompt, return_tensors="pt")
        prompt_ids = encoded.input_ids[0]
        answer_ids = tokenizer(
            " " + train_data["target"], add_special_tokens=False,
            return_tensors="pt").input_ids[0]
        if tokenizer.eos_token_id is not None:
            answer_ids = torch.cat([answer_ids, torch.tensor(
                [tokenizer.eos_token_id], dtype=answer_ids.dtype)])
        input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(model.device)
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_ids.numel():] = input_ids[:, prompt_ids.numel():]
        dtype = next(model.vision_model.parameters()).dtype
        pixel_values = encoded.pixel_values.to(model.device, dtype=dtype)
        attention_mask = torch.ones_like(input_ids)
        return model(pixel_values=pixel_values, input_ids=input_ids,
                     attention_mask=attention_mask, labels=labels,
                     return_dict=True).loss

    if is_qwen_vl(config.model_name):
        prompt = _one_word_prompt(train_data["prompt"])
        prompt_inputs = _qwen_inputs(
            processor, train_data["image"], prompt)
        full_inputs = _qwen_inputs(
            processor, train_data["image"], prompt, train_data["target"])
        prompt_len = prompt_inputs.input_ids.shape[1]
        full_inputs = full_inputs.to(model.device)
        labels = full_inputs.input_ids.clone()
        labels[:, :prompt_len] = -100
        return model(**full_inputs, labels=labels, use_cache=False,
                     return_dict=True).loss

    raise NotImplementedError(f"Token loss is not implemented for {config.model_name}")


def build_token_locality_loss_fn(config, model, tokenizer, processor, locality_data):
    """Cache pre-edit next-token distributions and return differentiable KL."""
    cached = []
    for item in locality_data:
        if is_qwen_vl(config.model_name):
            prompt = _one_word_prompt(item["prompt"])
            prompt_inputs = _qwen_inputs(
                processor, item["image"], prompt)
            full_inputs = _qwen_inputs(
                processor, item["image"], prompt, item["target"])
            prompt_len = prompt_inputs.input_ids.shape[1]
            kwargs = dict(full_inputs.to(model.device))
            labels = kwargs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            answer_mask = labels[:, 1:].ne(-100)
            with torch.no_grad():
                logits = model(**kwargs, return_dict=True).logits[:, :-1].float()
                reference_probs = F.softmax(
                    logits[answer_mask], dim=-1).detach()
            cached.append((kwargs, reference_probs, answer_mask))
            continue
        if config.model_name == "llava":
            image = Image.open(item["image"]).convert("RGB")
            dtype = next(model.model.mm_projector.parameters()).dtype
            images = process_images([image], processor, model.config).to(
                model.device, dtype=dtype)
            input_ids = tokenizer_image_token(
                item["prompt"], tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors="pt").unsqueeze(0).to(model.device)
            kwargs = {"input_ids": input_ids, "images": images}
        elif config.model_name == "blip2":
            image = Image.open(item["image"]).convert("RGB")
            inputs = processor(images=image, text=item["prompt"],
                               return_tensors="pt")
            dtype = next(model.vision_model.parameters()).dtype
            kwargs = {"pixel_values": inputs.pixel_values.to(model.device, dtype=dtype),
                      "input_ids": inputs.input_ids.to(model.device),
                      "attention_mask": inputs.attention_mask.to(model.device)}
        else:
            raise NotImplementedError(
                f"Token locality is not implemented for {config.model_name}")
        with torch.no_grad():
            logits = model(**kwargs, return_dict=True).logits[:, -1].float()
            reference_probs = F.softmax(logits, dim=-1).detach()
        cached.append((kwargs, reference_probs, None))

    def locality_kl():
        losses = []
        for kwargs, reference_probs, answer_mask in cached:
            logits = model(**kwargs, return_dict=True).logits.float()
            logits = (logits[:, -1] if answer_mask is None
                      else logits[:, :-1][answer_mask])
            losses.append(F.kl_div(F.log_softmax(logits, dim=-1),
                                   reference_probs, reduction="batchmean"))
        return (torch.stack(losses).mean() if losses else
                torch.zeros((), device=model.device))
    return locality_kl


def _select_locality_candidate(dataset, edit_index):
    """Choose an image-disjoint candidate with a different attribute."""
    current = dataset[edit_index]
    current_attr = current["metadata"]["attribute_type"]
    related = set(current["metadata"]["related_images"])
    for offset in range(1, len(dataset)):
        candidate = dataset[(edit_index + offset) % len(dataset)]["locality_candidate"]
        if (candidate["attribute_type"] != current_attr
                and candidate["image"] not in related
                and candidate["prompt"] and candidate["target"]):
            return candidate
    raise RuntimeError(
        f"No disjoint cross-attribute locality sample for edit {edit_index}")


def question_answer(config, d, processor, model, tokenizer):
    if config.model_name == 'llava':
        image = Image.open(d['image']).convert("RGB")
        image_tensor = process_images([image], processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            d['prompt'],
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )  # shape: [seq_len]
        input_ids = input_ids.unsqueeze(0).to(model.device, dtype=torch.long)  # shape: [1, seq_len]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                image_tensor,
                max_new_tokens=5,
                output_scores=True,
                return_dict_in_generate=True
            )

        pred = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()

    elif config.model_name == 'blip2':
        image = Image.open(d['image']).convert("RGB")

        inputs = processor(
            text=d['prompt'],
            images=image,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=5)

        predictions = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = predictions.split("Short answer:")[-1].strip()


    elif is_qwen_vl(config.model_name):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': d['image']}, {'type': 'text', 'text': d['prompt']}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(messages)
        inputs = processor(text=text,
                           images=img_in,
                           padding=True,
                           return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    return pred


def prediction(config, processor, model, tokenizer, data, result, mode):
    imgs, prompts, targets, pred_token = [], [], [], []

    for d in data:
        imgs.append(d['image']); prompts.append(d['prompt']); targets.append(d['target'])


        pred = question_answer(config, d, processor, model, tokenizer)
        pred_token.append(pred)
        # probs = [F.softmax(score[0], dim=-1) for score in generated_ids.scores]
        # generated_token_ids = generated_ids.sequences[0][1:]  # batch=0
        # token_probs = [prob[token_id].item() for prob, token_id in zip([F.softmax(score[0], dim=-1) for score in generated_ids.scores], generated_ids.sequences[0][1:])]

        # probs = [F.softmax(score[0], dim=-1) for score in generated_ids.scores][-1]
        # generated_token_ids = generated_ids.sequences[0][-1]
        # token_probs = probs[generated_token_ids].item()

        # pred_token.append((pred, np.prod(token_probs)))

    result['image'] = imgs
    result['prompt'] = prompts
    result['target'] = targets
    if mode=='pre_edit':
        result["pre_edit"] = pred_token
    if mode=='post_edit':
        result["post_edit"] = pred_token


def mean_result(results, samples_num):
    avg = {
        'samples_num': samples_num,
        'rel': 0.0,'loc_in': 0.0,'loc_out':0.0,
        'rephrase_image':0.0,'gen1':0.0,'gen2':0.0,
    }

    # import json
    # with open("records_all_layers.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    for result in results:
        for _key in result:
            if _key.startswith("loc_") or "Loc" in _key:
                result[_key]['acc'] = 1.0 if result[_key]['pre_edit']==result[_key]['post_edit'] else 0.0
            else:
                result[_key]['acc'] = 1.0 if result[_key]['target']==result[_key]['post_edit'] else 0.0
            avg[_key] += result[_key]['acc']

    for _key in avg:
        avg[_key] /= max(len(results), 1)

    avg['samples_num'] = len(results)

    return avg


def edit_model(config, processor, model, tokenizer, dataset,
               trg_model, context_model, val_loader, layernum):
    train_args = Namespace(
        model_name=config.model_name,
        ntrain=1,
        nsteps=config.nsteps,
        lr=config.lr,
        restrict_rank=config.restrict_rank,
        nsteps_proj=10,
        rank=config.rank,
        use_mask=True,
        feature_loss_weight=config.feature_loss_weight,
        token_loss_weight=config.token_loss_weight,
        token_locality_loss_weight=config.token_locality_loss_weight,
        spatial_merge_size=getattr(context_model, "spatial_merge_size", 1),
    )

    features = {}
    hooks = context_helpers._add_necessary_hooks(config, model, layernum=layernum, features=features)

    zm_k_path = (config.zm_k_path or os.path.join('zm_k', config.model_name, f'Vzm_k_{layernum}.pt'))
    os.makedirs(os.path.dirname(zm_k_path), exist_ok=True)
    if os.path.exists(zm_k_path):
        ZM_k = torch.load(zm_k_path)
    else:
        if config.model_name == "llava":
            _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.model.vision_tower, features, batch_size=2000, key_method='zca')
        elif config.model_name == "blip2":
            _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.vision_model, features, batch_size=2000, key_method='zca')
        elif is_qwen_vl(config.model_name):
            _, ZM_k = context_helpers.get_cov_matrix(
                val_loader, get_qwen_visual(model), features,
                batch_size=2000, key_method='zca')
        torch.save(ZM_k, zm_k_path)

    context_helpers._clear_specific_hooks(hooks)

    results = []
    edit_dataset = dataset[:config.sample_limit] if config.sample_limit else dataset
    for edit_index, data in enumerate(tqdm(edit_dataset)):
        result = {
            'rel': {},'loc_in': {},'loc_out':{},
            'rephrase_image':{},'gen1':{},'gen2':{},
        }

        target_state_backup = {k: v.detach().cpu().clone() for k, v in trg_model.state_dict().items()}

        # add hooks
        features = {}
        hooks = context_helpers._add_necessary_hooks(config, model, layernum=layernum, features=features)

        for data_key in data:
            # if data_key  == "loc":
            if data_key in ["loc", "gen"]:
                for _key in data[data_key]:
                    prediction(config, processor, model, tokenizer, data[data_key][_key], result[_key], mode="pre_edit")
            elif data_key == "rel":
                prediction(config, processor, model, tokenizer, data[data_key], result[data_key], mode="pre_edit")

        locality_item = _select_locality_candidate(dataset, edit_index)
        token_locality_loss_fn = build_token_locality_loss_fn(
            config, model, tokenizer, processor, [locality_item])
        context_model = rewrite_helpers.edit_classifier(
            train_args,
            data['inner'],
            context_model,
            ZM_k,
            features,
            target_model=trg_model,
            token_locality_loss_fn=token_locality_loss_fn,
            token_loss_fn=lambda: teacher_forcing_token_loss(
                config, model, tokenizer, processor, data['inner']),
        )

        for data_key in data:
            if data_key in ["loc", "gen"]:
                for _key in data[data_key]:
                    prediction(config, processor, model, tokenizer, data[data_key][_key], result[_key], mode="post_edit")
            elif data_key == "rel":
                prediction(config, processor, model, tokenizer, data[data_key], result[data_key], mode="post_edit")
        results.append(result)

        # remove hooks
        context_helpers._clear_specific_hooks(hooks)

        trg_model.load_state_dict(target_state_backup)
        del target_state_backup, data
        torch.cuda.empty_cache()
        gc.collect()

    avg_result = mean_result(results, len(dataset))

    return results, avg_result
