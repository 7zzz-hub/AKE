import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_model_root = os.environ.get("AKE_MODEL_ROOT", os.path.join(ROOT_PATH, "huggingface_cache"))
model_path_map = {
    "llava": os.environ.get("AKE_LLAVA_PATH", os.path.join(_model_root, "llava-1.5-7b-hf")),
    "blip2": os.environ.get("AKE_BLIP2_PATH", os.path.join(_model_root, "blip2-opt-6.7b")),
    "qwenvl": os.environ.get("AKE_QWENVL_PATH", os.path.join(_model_root, "Qwen3-VL-8B-Instruct")),
}
model_path_map.update({
    "llava-v1.5-7b": model_path_map["llava"],
    "blip2-opt-6.7b": model_path_map["blip2"],
})
