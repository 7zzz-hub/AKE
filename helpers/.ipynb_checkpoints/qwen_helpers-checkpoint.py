import torch


QWEN_VL_MODEL_NAMES = frozenset({"qwen2-vl", "qwen3-vl"})


def is_qwen_vl(model_name):
    return model_name in QWEN_VL_MODEL_NAMES


def get_qwen_visual(model):
    """Return the vision tower across Qwen2-VL and Qwen3-VL wrappers."""
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Could not locate the Qwen-VL vision tower")


def get_qwen_mlp_output(block):
    """Return the editable vision-MLP output projection."""
    if hasattr(block.mlp, "fc2"):
        return block.mlp.fc2
    if hasattr(block.mlp, "linear_fc2"):
        return block.mlp.linear_fc2
    raise AttributeError("Could not locate the Qwen-VL vision MLP output layer")


def mask_to_qwen_patch_order(mask, token_grid, merge_size):
    """Pool a spatial mask and match the processor's merge-window patch order."""
    temporal, grid_h, grid_w = map(int, token_grid)
    merge_size = int(merge_size)
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(
            f"Qwen grid {(temporal, grid_h, grid_w)} is not divisible by "
            f"spatial_merge_size={merge_size}")

    pooled = torch.nn.functional.adaptive_avg_pool2d(
        mask, output_size=(grid_h, grid_w))[0, 0]
    # Qwen image processors group patches by spatial-merge window before
    # flattening: [t, h//m, w//m, m, m]. Vision blocks preserve this order.
    return (pooled.unsqueeze(0)
            .expand(temporal, -1, -1)
            .reshape(temporal, grid_h // merge_size, merge_size,
                     grid_w // merge_size, merge_size)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1))
