from argparse import Namespace
import gc
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from helpers import rewrite_helpers, context_helpers
import torch.nn.functional as F

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def prediction(processor, model, tokenizer, data, result, mode):
    imgs, prompts, targets, pred_token = [], [], [], []

    for d in data:
        imgs.append(d['image']); prompts.append(d['prompt']); targets.append(d['target'])

        image = Image.open(d['image']).convert("RGB")
        image_tensor = process_images([image], processor.image_processor, model.config)
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
                max_new_tokens=3,
                output_scores=True,
                return_dict_in_generate=True
            )
    
        pred = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True).split('ASSISTANT: ')[-1]
        
        probs = [F.softmax(score[0], dim=-1) for score in generated_ids.scores]
        generated_token_ids = generated_ids.sequences[0][1:]  # batch=0
        token_probs = [prob[token_id].item() for prob, token_id in zip([F.softmax(score[0], dim=-1) for score in generated_ids.scores], generated_ids.sequences[0][1:])]

        probs = [F.softmax(score[0], dim=-1) for score in generated_ids.scores][-1]
        generated_token_ids = generated_ids.sequences[0][-1]
        token_probs = probs[generated_token_ids].item()

        pred_token.append((pred, np.prod(token_probs)))

    if mode=="pre_edit":
        result['image'] = imgs
        result['prompt'] = prompts
        result['target'] = targets
        result["pre_edit"] = pred_token
    elif mode=="post_edit":
        result["post_edit"] = pred_token
    

def mean_result(results, samples_num):
    avg = {
        'samples_num': samples_num,
        'rel': 0.0,'Loc_in': 0.0,'Loc_out':0.0,
        'rephrase_image':0.0,'gen1':0.0,'gen2':0.0,
    }

    for result in results:
        for _key in result:
            if "Loc" in _key:
                t = [a[0]==b[0] for a, b in zip(result[_key]['pre_edit'], result[_key]['post_edit'])]
                result[_key]['acc'] = sum(t)/len(t)
                avg[_key] += sum(t)/len(t) 
            elif "gen_3" not in _key:
                t = [a==b[0] for a, b in zip(result[_key]['target'], result[_key]['post_edit'])]
                result[_key]['acc'] = sum(t)/len(t)
                avg[_key] += sum(t)/len(t)
            else: 
                t = [a[0] for a in result[_key]['post_edit']]
                result[_key]['acc'] = 1.0 if result[_key]['target']==t else 0.0
                avg[_key] += 1.0 if result[_key]['target']==t else 0.0
    
    for _key in avg:
        avg[_key] /= max(len(results), 1)

    return avg


def edit_model(processor, model, tokenizer, dataset, trg_model, context_model, val_loader, layernum):
    train_args = Namespace(
        ntrain=1,
        nsteps=5000,
        lr=1e-4,
        restrict_rank=True,
        nsteps_proj=10,
        rank=1,
        use_mask=True
    )

    features = {}
    hooks = context_helpers._add_necessary_hooks(model, layernum=layernum, features=features)
    _, ZM_k = context_helpers.get_cov_matrix(val_loader, model.model.vision_tower, features, batch_size=2000, key_method='zca')
    context_helpers._clear_specific_hooks(hooks)

    results = []

    for data in tqdm(dataset):
        result = {
            'rel': 0.0,'Loc_in': 0.0,'Loc_out':0.0,
            'rephrase_image':0.0,'gen1':0.0,'gen2':0.0,
        }

        state_backup = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        # add hooks
        features = {}
        hooks = context_helpers._add_necessary_hooks(model, layernum=layernum, features=features)

        for data_key in data:
            if data_key in ["loc", "gen"]:
                for _key in data[data_key]:
                    prediction(processor, model, tokenizer, data[data_key][_key], result[_key], mode="pre_edit")
            elif data_key == "rel":
                prediction(processor, model, tokenizer, data[data_key], result[data_key], mode="pre_edit")

        context_model = rewrite_helpers.edit_classifier(
            train_args,
            data['inner'],
            context_model,
            ZM_k,
            features,
            target_model=trg_model
        )

        for data_key in data:
            if data_key in ["loc", "gen"]:
                for _key in data[data_key]:
                    prediction(processor, model, tokenizer, data[data_key][_key], result[_key], mode="post_edit")
            elif data_key == "rel":
                prediction(processor, model, tokenizer, data[data_key], result[data_key], mode="post_edit")
        results.append(result)

        # remove hooks
        context_helpers._clear_specific_hooks(hooks)

        model.load_state_dict(state_backup)
        del state_backup, data
        torch.cuda.empty_cache()
        gc.collect()

    avg_result = mean_result(results, len(dataset))

    return results, avg_result
