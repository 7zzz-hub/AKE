import os
import sys

import torch
import torch as ch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm

import helpers.math_helpers as math
from helpers.qwen_helpers import (get_qwen_mlp_output, get_qwen_visual,
                                  is_qwen_vl, mask_to_qwen_patch_order)
from tools import tally, pbar, renormalize, imgviz  

def _clear_specific_hooks(hooks):

    for hook in hooks:
        hook.remove()
    hooks.clear()


def _add_necessary_hooks(config, model, layernum, features):
    
    hooks = []
    def hook_fc2(module, input, output):
        features['fc2_pre'] = input[0]   
        features['fc2_post'] = output    

    def hook_layernorm(module, input, output):
        features['layer_norm2_pre'] = input[0]
        
    def hook_output(module, input, output):
        features['output_post'] = output[0] if isinstance(output, tuple) else output

    if config.model_name == 'llava':
        hook1 = model.model.vision_tower.vision_tower.vision_model.encoder.layers[layernum].mlp.fc2.register_forward_hook(hook_fc2)
        hook2 = model.model.vision_tower.vision_tower.vision_model.encoder.layers[layernum].layer_norm2.register_forward_hook(hook_layernorm)
        hook3 = model.model.vision_tower.vision_tower.vision_model.encoder.layers[layernum].register_forward_hook(hook_output)
    elif config.model_name == 'blip2':
        hook1 = model.vision_model.encoder.layers[layernum].mlp.fc2.register_forward_hook(hook_fc2)
        hook2 = model.vision_model.encoder.layers[layernum].layer_norm2.register_forward_hook(hook_layernorm)
        hook3 = model.vision_model.encoder.layers[layernum].register_forward_hook(hook_output)
    elif is_qwen_vl(config.model_name):
        visual = get_qwen_visual(model)
        block = visual.blocks[layernum]
        hook1 = get_qwen_mlp_output(block).register_forward_hook(hook_fc2)
        hook2 = block.norm2.register_forward_hook(hook_layernorm)
        hook3 = block.register_forward_hook(hook_output)

    # hooks.append(hook1)
    hooks.extend([hook1, hook2, hook3])

    return hooks
    

def get_keys(batch, features, context_mod=None, device='cuda', 
             no_grad=True, loc='input'):
    
    
    def get_keys_sub():
        # context_mod(batch['pixel_values'], grid_thw=batch['grid_thw'])
    
        if isinstance(batch, dict):
            dtype = next(context_mod.parameters()).dtype
            context_mod(batch['pixel_values'].to(device=device, dtype=dtype),
                        grid_thw=batch['image_grid_thw'].to(device))
        else:
            assert len(batch.shape) == 4
            context_mod(batch.to(device))

        if loc == 'input':
            if type(features['fc2_pre']) == tuple:
                return (features['fc2_pre'][0].detach().clone(), features['fc2_pre'][1].detach().clone())
            else:
                return features['fc2_pre'].detach().clone() 
        else:
            return features['fc2_post'].detach().clone()
    
    if no_grad:
        with torch.no_grad():
            return get_keys_sub()
    else:
        return get_keys_sub()

def get_cov_matrix(loader, context_model, features, batch_size=78400, 
                   key_method='zca', device='cuda', caching_dir=None,
                   force_recache=False):
   
    if caching_dir:
        paths = [os.path.join(caching_dir, p) 
                 for p in ['CM_k.pt', 'ZM_k.pt']]
        if all(os.path.exists(p) for p in paths) and not force_recache:
            print("Found precomputed cov matrices, returning...")
            ret = []
            for f in paths:
                ret.append(ch.load(f).to(device))
            return ret
                  
    print("Computing cov matrices...")
    CM_k = calculate_2nd_moment(loader, context_model, features,
                                       batch_size=batch_size, device=device)
    
    assert not ch.any(ch.isnan(CM_k)) 
    
    if key_method == 'zca':
        dtype = CM_k.dtype
        
        if not math.is_PD(CM_k.cpu().numpy()):
            print("Making CM_k PD")
            eps = 1e-6
            CM_k += eps * torch.eye(CM_k.shape[0], device=CM_k.device)
        assert math.is_PD(CM_k.cpu().numpy()) 

        ZM_k = math.zca_from_cov(CM_k).to(device)
        assert not ch.any(ch.isnan(ZM_k)) 
    else:
        ZM_k = ch.eye(CM_k.shape[0]).to(device)
     
    if caching_dir:
        paths = [os.path.join(caching_dir, p) 
                 for p in ['CM_k.pt', 'ZM_k.pt']]
        os.makedirs(caching_dir, exist_ok=True)
        for t, p in zip([CM_k, ZM_k], paths):
            ch.save(t, p)
    
    return CM_k, ZM_k

def calculate_2nd_moment(val_loader, context_model, features,
                                batch_size=78400, device='cuda'):
                      
    total_count = 0
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        zbatch = batch[0] if isinstance(batch, (tuple, list)) else batch
        acts = get_keys(zbatch, features,
                        context_mod=context_model, 
                        device=device)
        
        if type(acts) == tuple:
            acts = acts[0]

        if isinstance(acts, torch.Tensor):
            if acts.dim() == 3:
                sep_pix = acts.reshape(-1, acts.shape[2])
            elif acts.dim() == 2:
                sep_pix = acts
        
        if batch_idx == 0:
            moment = torch.zeros(
                (sep_pix.shape[1], sep_pix.shape[1]),
                device=sep_pix.device,
                dtype=torch.float64,
            )

        BC = int(np.ceil(sep_pix.shape[0] / batch_size))
        for iidx in range(BC):
            block = sep_pix[iidx * batch_size:(iidx + 1) * batch_size].double()
            moment.addmm_(block.t(), block)

        total_count += sep_pix.shape[0]
        assert not torch.any(torch.isnan(moment)), "Moment contains NaNs!"

    if total_count == 0:
        raise ValueError("Cannot compute a second moment from an empty loader")
    moment.div_(total_count)
    return moment.float()



def get_matches(context_k, ims, features, context_model, K=200, q=0.99):
    match_idx, match_im, match_mask, match_over = find_context_matches(
                                                       context_k, ims,
                                                       features, 
                                                       context_model, 
                                                       k=K, 
                                                       q=q)
    
    nz_mask = np.where(np.sum(match_mask.cpu().numpy().reshape(match_mask.shape[0], -1), axis=1) != 0)[0]
    match_idx, match_im, match_mask, match_over = (match_idx[nz_mask], match_im[nz_mask], match_mask[nz_mask],
                                                   match_over[nz_mask])
    
    return match_idx, match_mask, match_over
    
    
def get_context_key(source_imgs,
                     source_masks,
                     context_model,
                     matrix,
                     features,
                     rank=1,
                     device='cuda',
                     loc='input',
                     threshold=0.2):
    # Fairly ok
    with torch.no_grad():
        accumulated_obs = []
        is_qwen = isinstance(source_imgs, dict)
        image_batches = [source_imgs] if is_qwen else [img[None, ...] for img in source_imgs]
        for image_batch, mask in zip(image_batches, source_masks):
            k_acts = get_keys(image_batch, features, context_mod=context_model,
                              device=device, loc=loc)
            if type(k_acts) == tuple:
                k_acts = k_acts[0]

            token_count = k_acts.shape[0] if k_acts.dim() == 2 else k_acts.shape[1]
            num_patches = token_count if is_qwen else token_count - 1
            if is_qwen:
                grid = image_batch['image_grid_thw'][0].tolist()
                temporal, grid_h, grid_w = map(int, grid)
                if temporal * grid_h * grid_w != num_patches:
                    raise ValueError(
                        f'Qwen token grid {grid} does not match {num_patches} visual tokens')
                area_patch = mask_to_qwen_patch_order(
                    mask[None].float(), grid,
                    context_model.spatial_merge_size).reshape(-1, 1)
            else:
                patch_size = int(num_patches**0.5)
                if patch_size * patch_size != num_patches:
                    raise ValueError(
                        f'Expected a square image token grid, got {num_patches} tokens')
                area_patch = F.adaptive_avg_pool2d(
                    mask[None].float(), (patch_size, patch_size))[0, 0].reshape(-1, 1)
            area = area_patch.cuda() if is_qwen else torch.cat([
                torch.zeros((1, 1), device=area_patch.device), area_patch], dim=0).cuda()

            accumulated_obs.append((k_acts.reshape(-1, k_acts.shape[-1]), area))
        
        all_obs = torch.cat([obs[(w > 0).nonzero()[:, 0], :]
                             for obs, w in accumulated_obs])
        all_weight = torch.cat([w[w > 0]
                                for _, w in accumulated_obs])
        all_zca_k = torch.cat([(w * math.zca_whitened_query_key(matrix, obs))[(w > 0).nonzero()[:, 0], :]
                                for obs,  w in accumulated_obs])

        _, _, q = all_zca_k.svd(compute_uv=True)
        top_e_vec = q[:, :rank]
        # Map SVD directions out of whitened coordinates before projection.
        row_dirs = math.zca_unwhitened_direction(matrix, top_e_vec.t())
        just_avg = (all_zca_k).sum(0)
        q, r = torch.qr(row_dirs.permute(1, 0))
        signs = (q * just_avg[:, None]).sum(0).sign()
        q = q * signs[None, :]
        return q.permute(1, 0)
    
def find_context_matches(key, ims, features, context_model, k=12,  
                         device='cuda', loc='input', q=0.999):
    sel_idx, sel_imgs, query_rq = rank_using_context(key, ims, features, context_model, 
                                       k=k, device=device, loc=loc)    
    level = query_rq.quantiles(q)[0]
    masks, masked_imgs = find_matching_region_img(context_model,
                                            sel_imgs,
                                            key, 
                                            level,
                                            device=device,
                                            loc=loc,
                                            border_color=[255, 255, 255])
    return sel_idx, sel_imgs, masks, masked_imgs

def rank_using_context(key, ims, features, context_model, k=12, 
                       device='cuda', loc='input'):
    tensorkey = key.to(device).unsqueeze(2).unsqueeze(3)
    with pbar.quiet(), torch.no_grad():
        def image_max_sel(zbatch):
            acts = get_keys(zbatch, features,
                                context_mod=context_model, 
                                device=device,
                                loc=loc)
            if type(acts) == tuple:
                acts = acts[0]
            heatmap = (acts * tensorkey).sum(dim=1)
            maxmap = heatmap.view(heatmap.shape[0], -1).max(1)[0]
            flatmap = heatmap.view(-1)[:, None]
            return maxmap, flatmap
        topk, rq = tally.tally_topk_and_quantile(
            image_max_sel, ims, k=k)
    sel_idx = topk.result()[1]
    return sel_idx, ims[sel_idx], rq


def find_matching_region_img(context_model, imgs, key, level, 
                             device='cuda', loc='input', **kwargs):
        batch_size = 3
        masks, masked_imgs = [], []
        for i in range(0, len(imgs), batch_size):
            img_batch = imgs[i:i + batch_size]
            
            with torch.no_grad():
                tensorkey = key.to(device).unsqueeze(2).unsqueeze(3)
                acts = get_keys(img_batch, 
                                context_mod=context_model, 
                                device=device, loc=loc)
                if type(acts) == tuple:
                    acts = acts[0]
                heatmap = (acts[...] * tensorkey).sum(dim=1)

                imgdata_batch = 2 * (img_batch - 0.5)
                iv = imgviz.ImageVisualizer(imgdata_batch.shape[2:])
                
                
                masks.extend([iv.pytorch_mask(h, unit=None, level=level,
                                     percent_level=None).cpu().float()
                                for h in heatmap])
                
                masked_imgs.extend(
                    [iv.masked_image(imgdata, heatmap[j], level=level,
                                     **kwargs)
                     for j, imgdata in enumerate(imgdata_batch)])

        masked_imgs = ch.stack([ch.tensor(np.asarray(r)).permute(2, 0, 1) for r in masked_imgs])
        return ch.stack(masks).cpu(), masked_imgs