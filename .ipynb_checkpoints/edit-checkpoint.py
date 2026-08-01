from argparse import Namespace
import gc
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from helpers import rewrite_helpers, context_helpers
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def teacher_forcing_token_loss(config, model, tokenizer, train_data):
    """Return LLaVA loss while supervising answer tokens only."""
    if config.model_name != 'llava':
        raise NotImplementedError(
            'Teacher-forcing token loss is currently implemented for LLaVA only')

    prompt = (
        f"USER: {DEFAULT_IMAGE_TOKEN}\n {train_data['prompt']} "
        "Please answer in one word. ASSISTANT:"
    )
    prompt_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    answer_ids = tokenizer(
        " " + train_data['target'],
        add_special_tokens=False,
        return_tensors='pt',
    ).input_ids[0]
    if tokenizer.eos_token_id is not None:
        answer_ids = torch.cat([
            answer_ids,
            torch.tensor([tokenizer.eos_token_id], dtype=answer_ids.dtype),
        ])

    input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(model.device)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.numel():] = input_ids[:, prompt_ids.numel():]
    projector_dtype = next(model.model.mm_projector.parameters()).dtype
    images = train_data['imgs'][:1].to(model.device, dtype=projector_dtype)
    return model(
        input_ids=input_ids, labels=labels, images=images,
        use_cache=False, return_dict=True,
    ).loss

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
            outputs = model.generate(**inputs, do_sample=False)

        predictions = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = predictions.split("Short answer:")[-1].strip()


    elif config.model_name == 'qwen2-vl':
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
        pred = processor.decode(generated_ids_trimmed,skip_special_tokens=True, clean_up_tokenization_spaces=False,)

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
        'rel': 0.0,'Loc_m': 0.0,'Loc_t':0.0,
        'rephrase_image':0.0,'gen1':0.0,'gen2':0.0,
    }

    # import json
    # with open("records_all_layers.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)

    for result in results:
        for _key in result:
            if "Loc" in _key:
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
        lr=1e-3,
        restrict_rank=config.restrict_rank,
        nsteps_proj=10,
        rank=config.rank,
        use_mask=True,
        feature_loss_weight=config.feature_loss_weight,
        token_loss_weight=config.token_loss_weight,
        image_locality_loss_weight=config.image_locality_loss_weight,
    )

    features = {}
    hooks = context_helpers._add_necessary_hooks(config, model, layernum=layernum, features=features)

    zm_k_path = os.path.join('zm_k', config.model_name, f'Vzm_k_{layernum}.pt')
    os.makedirs(os.path.dirname(zm_k_path), exist_ok=True)
    if os.path.exists(zm_k_path):
        ZM_k = torch.load(zm_k_path)
    else:
        if config.model_name == "llava":
            _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.model.vision_tower, features, batch_size=2000, key_method='zca')
        elif config.model_name == "blip2":
            _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.vision_model, features, batch_size=2000, key_method='zca')
        elif config.model_name == 'qwen2-vl':
            _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.visual, features, batch_size=2000, key_method='zca')
        torch.save(ZM_k, zm_k_path)

    context_helpers._clear_specific_hooks(hooks)

    results = []
    locality_iter = iter(val_loader)

    for data in tqdm(dataset[:10]):
        result = {
            'rel': {},'Loc_m': {},'Loc_t':{},
            'rephrase_image':{},'gen1':{},'gen2':{},
        }

        state_backup = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

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

        try:
            locality_batch = next(locality_iter)
        except StopIteration:
            locality_iter = iter(val_loader)
            locality_batch = next(locality_iter)
        locality_images = (
            locality_batch[0] if isinstance(locality_batch, (tuple, list))
            else locality_batch)
        with torch.no_grad():
            context_model(locality_images.cuda())
        locality_pair = (features['fc2_pre'].detach().clone(),
                         features['fc2_post'].detach().clone())

        context_model = rewrite_helpers.edit_classifier(
            train_args,
            data['inner'],
            context_model,
            ZM_k,
            features,
            target_model=trg_model,
            token_loss_fn=lambda: teacher_forcing_token_loss(
                config, model, tokenizer, data['inner']),
            locality_pair=locality_pair,
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

        model.load_state_dict(state_backup)
        del state_backup, data
        torch.cuda.empty_cache()
        gc.collect()

    avg_result = mean_result(results, len(dataset))

    return results, avg_result
