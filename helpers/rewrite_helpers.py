import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import helpers.context_helpers as coh
from helpers.qwen_helpers import is_qwen_vl, mask_to_qwen_patch_order
from tools import renormalize, nethook


def downscale_mask(args, mask, tgt_size, threshold=None, token_grid=None):

    is_qwen = is_qwen_vl(args.model_name)
    patch_num = tgt_size if is_qwen else tgt_size - 1
    if is_qwen:
        if token_grid is None:
            raise ValueError('Qwen-VL mask downsampling requires image_grid_thw')
        temporal, grid_h, grid_w = map(int, token_grid)
        if temporal * grid_h * grid_w != patch_num:
            raise ValueError(
                f'Qwen token grid {token_grid} does not match {patch_num} tokens')
        pooled = mask_to_qwen_patch_order(
            mask, token_grid, args.spatial_merge_size)
    else:
        patch_size = int(patch_num**0.5)
        pooled = F.adaptive_avg_pool2d(
            mask, output_size=(patch_size, patch_size))[0, 0]

    if threshold is not None:
        pooled = pooled > threshold

    final_mask = pooled.flatten()
    if not is_qwen:
        cls_mask = torch.zeros(1, device=mask.device)
        final_mask = torch.cat([cls_mask, final_mask])  # shape: [num_tokens]

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
                           unfold=False, mask=None, token_loss_fn=None,
                           token_locality_loss_fn=None,
                           token_grid=None):

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

    def compute_feature_loss(mask=None):
        target = val
        prediction = target_model(key)
        if mask is not None:
            token_weights = downscale_mask(
                args, mask, target.shape[-2], None,
                token_grid=token_grid).sqrt()
            token_weights = token_weights.to(
                device=prediction.device, dtype=prediction.dtype)
            weight_shape = [1] * prediction.dim()
            weight_shape[-2] = token_weights.numel()
            weights = token_weights.view(weight_shape)
            absolute_error = torch.abs(prediction - target)
            normalizer = weights.sum() * absolute_error.shape[0] * absolute_error.shape[-1]
            return (absolute_error * weights).sum() / normalizer.clamp_min(1e-8)
        return F.l1_loss(prediction, target)

    # set up optimizer
    weight = target_weights(target_model)
    weight_orig = weight.clone()
    params = [weight]
    if low_rank_insert or low_rank_gradient:
        with torch.no_grad():
            ortho_weight = weight - projected_conv(weight, context)

    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-4)

    pbar = tqdm(range(niter))
    for it in pbar:
        with torch.enable_grad():
            feature_loss = compute_feature_loss(mask)
            token_loss = (token_loss_fn() if token_loss_fn is not None
                          else feature_loss.new_zeros(()))
            token_locality_loss = (
                token_locality_loss_fn() if token_locality_loss_fn is not None
                else feature_loss.new_zeros(()))
            loss = (args.feature_loss_weight * feature_loss
                    + args.token_loss_weight * token_loss
                    + args.token_locality_loss_weight * token_locality_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite edit loss at step {it}: {loss.item()}")
            optimizer.zero_grad()
            if it in (0, niter // 2, niter - 1):
                feature_grad = torch.autograd.grad(
                    feature_loss, weight, retain_graph=True,
                    allow_unused=True)[0]
                token_grad = torch.autograd.grad(
                    token_loss, weight, retain_graph=True,
                    allow_unused=True)[0]
                feature_norm = (feature_grad.norm().item()
                                if feature_grad is not None else 0.0)
                token_norm = (token_grad.norm().item()
                              if token_grad is not None else 0.0)
                if token_grad is not None:
                    projected_token_norm = projected_conv(
                        token_grad, context).norm().item()
                    retained = projected_token_norm / max(token_norm, 1e-12)
                else:
                    projected_token_norm, retained = 0.0, 0.0
                print(
                    f'Gradient diagnostics step={it}: '
                    f'feature={feature_norm:.6g}, token={token_norm:.6g}, '
                    f'token_projected={projected_token_norm:.6g}, '
                    f'retained={retained:.4%}')
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
    print("Final feature/token/token-locality loss:", feature_loss.item(),
          token_loss.item(), token_locality_loss.item())
    print("L2 norm of weight change:", torch.norm(weight_orig - weight).item())

def edit_classifier(args, train_data,
               context_model,
               ZM_k,
               features,
               target_model=None,
               caching_dir=None,
               token_loss_fn=None,
               token_locality_loss_fn=None):

    assert args.ntrain <= len(train_data['imgs'])

    # batch
    is_qwen = isinstance(train_data['imgs'], dict)
    if not is_qwen:
        train_data['imgs'] = train_data['imgs'].unsqueeze(0)
        train_data['modified_imgs'] = train_data['modified_imgs'].unsqueeze(0)
    train_data['masks'] = train_data['masks'].unsqueeze(0)

    if is_qwen:
        if not torch.equal(train_data["imgs"]["image_grid_thw"],
                           train_data["modified_imgs"]["image_grid_thw"]):
            raise ValueError(
                "Original and edited Qwen images must produce the same image_grid_thw")
        cp_imgs = {k: torch.cat([train_data['imgs'][k], train_data['modified_imgs'][k]]) for k in ('pixel_values', 'image_grid_thw')}
        cp_imgs['pixel_values'] = cp_imgs['pixel_values'].float()
        token_grid = train_data['imgs']['image_grid_thw'][0].tolist()
    else:
        cp_imgs = torch.cat([train_data['imgs'][:args.ntrain], train_data['modified_imgs'][:args.ntrain]]).float()
        token_grid = None
    cp_masks = torch.cat([train_data['masks'][:args.ntrain],
                       train_data['masks'][:args.ntrain]]).float()

    Nims = 2 if is_qwen else len(cp_imgs)

    assert (target_model is not None) and (ZM_k is not None)


    context_k = coh.get_context_key(train_data['modified_imgs'] if is_qwen else train_data['modified_imgs'].float(),
                                        train_data['masks'],
                                        context_model, ZM_k, features,
                                        rank=args.rank)
    context_k = context_k.to(device=target_weights(target_model).device,
                             dtype=target_weights(target_model).dtype)

    with torch.no_grad():
        if is_qwen:
            context_model(cp_imgs['pixel_values'].to(device='cuda', dtype=target_weights(target_model).dtype),
                          grid_thw=cp_imgs['image_grid_thw'].cuda())
        else:
            context_model(cp_imgs.cuda())

    # kstar = features['fc2_pre'][Nims//2:].detach().clone()
    # vstar = (features['output_post'][:Nims//2] - features['layer_norm2_pre'][Nims//2:]).detach().clone()

    if is_qwen and features['fc2_pre'].shape[0] % 2:
        raise ValueError('Original and edited Qwen images produced unequal token counts')
    split = features['fc2_pre'].shape[0] // 2 if is_qwen else Nims // 2
    kstar = features['fc2_pre'][:split].detach().clone()
    # vstar = (features['output_post'][split:] - features['layer_norm2_pre'][:split]).detach().clone()
    vstar = (features['output_post'][split:] - features['layer_norm2_pre'][:split]).detach().clone()

    mstar = torch.max(cp_masks[:Nims//2], dim=1, keepdims=True)[0]

    edit_classifier_weights(args, target_model, kstar, vstar,
                                   context_k, niter=args.nsteps,
                                   piter=args.nsteps_proj, lr=args.lr,
                                   low_rank_insert=args.restrict_rank,
                                   mask=mstar.cuda() if args.use_mask else None,
                                   token_loss_fn=token_loss_fn,
                                   token_locality_loss_fn=token_locality_loss_fn,
                                   token_grid=token_grid)

    return context_model
