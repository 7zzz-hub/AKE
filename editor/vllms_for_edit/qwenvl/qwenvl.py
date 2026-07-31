import inspect
from typing import List, Optional

import torch
from PIL import Image, ImageOps
from PIL.Image import Image as ImageClass
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base import BaseVLLMForEdit


class QwenVLForEdit(BaseVLLMForEdit):
    """Qwen3-VL wrapper used by VEAD.

    Images are padded to a fixed resolution because the VEAD adaptor requires a
    stable number of visual tokens. Qwen3-VL DeepStack features are preserved
    and injected into the text decoder rather than discarded.
    """

    IMAGE_SIZE = 672
    IMAGE_TOKEN = "<|image_pad|>"

    def __init__(self, model_path: str, device: str = "cuda", *args, **kwargs):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL requires transformers>=4.57.0. "
                "Please upgrade transformers before loading qwen3vl."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = kwargs.pop("torch_dtype", torch.float32)
        load_kwargs = {"trust_remote_code": True, "dtype": dtype}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.device = next(self.model.parameters()).device
        super().__init__(self.model, self.device, False)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def _language_model(self):
        return self.model.model.language_model

    def _visual_model(self):
        return self.model.model.visual

    def get_img_token_n(self):
        cfg = self.model.config.vision_config
        patch = int(getattr(cfg, "patch_size", 16))
        merge = int(getattr(cfg, "spatial_merge_size", 2))
        return (self.IMAGE_SIZE // patch // merge) ** 2

    def is_q_former_based(self):
        return False

    def _build_prompts(self, texts: List[str], imgs: List[ImageClass]):
        prompts = []
        for text, img in zip(texts, imgs):
            if self.IMAGE_TOKEN in text:
                if text.count(self.IMAGE_TOKEN) != 1:
                    raise ValueError("Each Qwen3-VL prompt must contain exactly one image token.")
                prompts.append(text)
                continue
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ],
            }]
            prompts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return prompts

    @staticmethod
    def _flatten_image_features(features):
        if isinstance(features, (tuple, list)):
            return torch.cat(list(features), dim=0)
        if features.dim() == 3:
            return features.reshape(-1, features.shape[-1])
        return features

    @staticmethod
    def _unpack_image_features(output):
        """Support both the 4.57 tuple API and older ModelOutput APIs."""
        if isinstance(output, (tuple, list)):
            image_features = output[0]
            deepstack_features = output[1] if len(output) > 1 else None
        else:
            image_features = getattr(output, "pooler_output", None)
            if image_features is None:
                image_features = getattr(output, "last_hidden_state", None)
            deepstack_features = getattr(output, "deepstack_features", None)
        if image_features is None:
            raise TypeError(
                "Unsupported Qwen-VL image feature output: "
                f"{type(output).__name__}"
            )
        return image_features, deepstack_features

    def _position_ids(self, input_ids, attention_mask, image_grid_thw, mm_token_type_ids):
        get_rope_index = self.model.model.get_rope_index
        rope_kwargs = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": None,
            "attention_mask": attention_mask,
        }
        # transformers 4.57 derives multimodal positions from the special token
        # IDs, while newer Qwen3-VL implementations also accept explicit types.
        if "mm_token_type_ids" in inspect.signature(get_rope_index).parameters:
            rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
        position_ids, _ = get_rope_index(**rope_kwargs)
        return position_ids

    def get_llm_input_embeds(
        self, texts: List[str], imgs: Optional[List[ImageClass]] = None
    ):
        embed_layer = self.model.get_input_embeddings()
        embed_device = embed_layer.weight.device

        if imgs is None:
            encoded = self.get_llm_tokenizer()(
                texts, return_tensors="pt", padding=True, return_attention_mask=True
            )
            input_ids = encoded.input_ids.to(embed_device)
            return {
                "inputs_embeds": embed_layer(input_ids),
                "attention_mask": encoded.attention_mask.to(embed_device),
                "position_ids": None,
                "visual_pos_masks": None,
                "deepstack_visual_embeds": None,
            }, None

        if len(texts) != len(imgs):
            raise ValueError(f"Got {len(texts)} texts but {len(imgs)} images.")
        imgs = [ImageOps.pad(
            img.convert("RGB"), (self.IMAGE_SIZE, self.IMAGE_SIZE),
            method=Image.Resampling.BICUBIC, color=(0, 0, 0)
        ) for img in imgs]
        prompts = self._build_prompts(texts, imgs)
        encoded = self.processor(
            text=prompts,
            images=imgs,
            return_tensors="pt",
            padding=True,
            return_mm_token_type_ids=True,
        )

        input_ids = encoded["input_ids"].to(embed_device)
        attention_mask = encoded["attention_mask"].to(embed_device)
        mm_token_type_ids = encoded.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(embed_device)
        image_grid_thw = encoded["image_grid_thw"].to(embed_device)
        inputs_embeds = embed_layer(input_ids)

        visual = self._visual_model()
        visual_device = next(visual.parameters()).device
        visual_dtype = next(visual.parameters()).dtype
        vision_output = self.model.get_image_features(
            pixel_values=encoded["pixel_values"].to(visual_device, dtype=visual_dtype),
            image_grid_thw=encoded["image_grid_thw"].to(visual_device),
        )
        image_features, deepstack = self._unpack_image_features(vision_output)
        image_embeds = self._flatten_image_features(image_features)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_token_mask = input_ids.eq(self.get_img_special_token_id())
        if image_token_mask.sum().item() != image_embeds.shape[0]:
            raise ValueError(
                "Qwen3-VL image token/feature mismatch: "
                f"{image_token_mask.sum().item()} tokens vs {image_embeds.shape[0]} features"
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            image_token_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
        )

        positions = torch.where(image_token_mask[0])[0]
        if positions.numel() == 0:
            raise ValueError("Qwen3-VL processor did not create image placeholder tokens.")
        vt_range = [int(positions[0]), int(positions[-1]) + 1]
        if vt_range[1] - vt_range[0] != self.get_img_token_n():
            raise ValueError(
                f"Expected {self.get_img_token_n()} visual tokens at fixed resolution, "
                f"but got {vt_range[1] - vt_range[0]}."
            )

        if deepstack is not None:
            deepstack = [x.to(inputs_embeds.device, inputs_embeds.dtype) for x in deepstack]
        llm_input = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": self._position_ids(
                input_ids, attention_mask, image_grid_thw, mm_token_type_ids
            ),
            "visual_pos_masks": image_token_mask,
            "deepstack_visual_embeds": deepstack,
        }
        return llm_input, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range=None):
        kwargs = {
            "inputs_embeds": llm_inpt["inputs_embeds"],
            "attention_mask": llm_inpt["attention_mask"],
            "use_cache": False,
            "return_dict": True,
        }
        for key in ("position_ids", "visual_pos_masks", "deepstack_visual_embeds"):
            if llm_inpt.get(key) is not None:
                kwargs[key] = llm_inpt[key]
        output = self._language_model()(**kwargs)
        return CausalLMOutputWithPast(
            logits=self.model.lm_head(output.last_hidden_state),
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def get_img_special_token_str(self):
        # Returning None lets raw prompts reach _build_prompts; the Qwen chat
        # template inserts the complete vision-start/image-pad/vision-end span.
        return None

    def get_img_special_token_id(self):
        return self.model.config.image_token_id
