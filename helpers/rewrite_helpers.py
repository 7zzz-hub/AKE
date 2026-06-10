import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import helpers.context_helpers as coh
from tools import renormalize, nethook


def downscale_mask(args, mask, tgt_size, threshold=None):

    patch_num = args.patch_num - 1  # exclude CLS token
    patch_size = int(patch_num**0.5) 

    pooled = F.adaptive_avg_pool2d(mask, output_size=(patch_size, patch_size))[0,0]

    if threshold is not None:
        pooled = pooled > threshold

    cls_mask = torch.zeros(1, device=mask.device)
    final_mask = torch.cat([cls_mask, pooled.flatten()])  # shape: [num_tokens]

    return final_mask

def target_weights(target_model):
    return [p for n, p in target_model.named_parameters()
            if 'weight' in n][0]
    
def projected_conv(weight, direction):
    cosine_map = torch.einsum('oi, di -> od', weight, direction)  # [out_dim, num_directions]
    result = torch.einsum('od, di -> oi', cosine_map, direction)  # [out_dim, in_dim]
    
    return result

def edit_classifier_weights(args, target_model, key, val, context, 
                           niter=2001, piter=10, lr=0.05, 
                           low_rank_insert=True, low_rank_gradient=False,
                           unfold=False, mask=None):
    
    def update_callback(it, loss, pbar=None):
        if it % 50 == 0 or it == niter - 1:
            loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                         f'\tloss {loss.item():.4f}')
            if pbar:
                pbar.set_description(str(loss))
            else:
                print(loss_info)
    try:
        key, val = [d.detach() for d in [key, val]]
    except:
        val = val.detach()

    def compute_loss(args, mask=None):
        reps = val, target_model(key)
        if mask is not None:
            mask = downscale_mask(args, mask, val.shape[-1], None)
            mask = mask.sqrt()
            reps = [r * mask.unsqueeze(1) for r in reps]
        return torch.nn.functional.l1_loss(*reps) / len(val)

    # set up optimizer
    weight = target_weights(target_model)
    weight_orig = weight.clone()
    params = [weight]
    if low_rank_insert or low_rank_gradient:
        with torch.no_grad():
            ortho_weight = weight - projected_conv(weight, context)
            
    optimizer = torch.optim.Adam(params, lr=lr)

    pbar = tqdm(range(niter))
    for it in pbar:
        with torch.enable_grad():
            loss = compute_loss(mask)
            optimizer.zero_grad()
            loss.backward()

            if it == 0: loss_orig = loss.item()

            if low_rank_gradient:
                weight.grad[...] = projected_conv(weight.grad, context)
            optimizer.step()
            if update_callback is not None:
                update_callback(it, loss, pbar=pbar)
            if low_rank_insert and (it % piter == 0 or it == niter - 1):
                with torch.no_grad():
                    weight[...] = (
                        ortho_weight + projected_conv(weight, context))

    print("Loss (orig, final):", loss_orig, loss.item())
    print("L2 norm of weight change:", torch.norm(weight_orig - weight).item())
    
def edit_classifier(args, train_data, 
               context_model, 
               ZM_k,
               features,
               target_model=None,   
               caching_dir=None):
                
    assert args.ntrain <= len(train_data['imgs'])

    # batch
    train_data['imgs'] = train_data['imgs'].unsqueeze(0)
    train_data['modified_imgs'] = train_data['modified_imgs'].unsqueeze(0)
    train_data['masks'] = train_data['masks'].unsqueeze(0)

    cp_imgs = torch.cat([train_data['imgs'][:args.ntrain], 
                      train_data['modified_imgs'][:args.ntrain]]).float()
    cp_masks = torch.cat([train_data['masks'][:args.ntrain], 
                       train_data['masks'][:args.ntrain]]).float()
    
    Nims = len(cp_imgs)
    
    assert (target_model is not None) and (ZM_k is not None)

    
    context_k = coh.get_context_key(train_data['modified_imgs'].float(), 
                                        train_data['masks'], 
                                        context_model, ZM_k, features,
                                        rank=args.rank)
    
    with torch.no_grad(): context_model(cp_imgs.cuda())

    kstar = features['fc2_pre'][Nims//2:].detach().clone()
    # vstar = (features['layer_norm2_pre'][:Nims//2] - features['layer_norm2_pre'][Nims//2:] + features['fc2_post'][:Nims//2]).detach().clone()
    vstar = (features['output_post'][:Nims//2] - features['layer_norm2_pre'][Nims//2:]).detach().clone()
        
    mstar = torch.max(cp_masks[:Nims//2], dim=1, keepdims=True)[0]
        
    edit_classifier_weights(args, target_model, kstar, vstar, 
                                   context_k, niter=args.nsteps, 
                                   piter=args.nsteps_proj, lr=args.lr, 
                                   low_rank_insert=args.restrict_rank, 
                                   mask=mstar.cuda() if args.use_mask else None)
       
    return context_model