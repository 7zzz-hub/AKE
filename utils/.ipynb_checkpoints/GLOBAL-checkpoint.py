# ROOT_PATH = 'VEAD'
# model_path_map = {
#     'llava-v1.5-7b': 'models/llava-v1.5-7b-hf',
#     'blip2-opt-2.7b': 'models/blip2-opt-2.7b',
#     'minigpt-4-vicuna-7b': 'models/minigpt-4-vicuna-7b',
# }

ROOT_PATH = '.'
model_path_map = {
    # Will be auto-downloaded from Hugging Face if not present locally
    'llava': '../huggingface_cache/llava-1.5-7b-hf',
    'blip2': '/root/autodl-tmp/VisEdit/huggingface_cache/blip2-opt-6.7b',
    'qwenvl': '/root/autodl-tmp/VisEdit/huggingface_cache/Qwen2-VL'
}