from typing import List, Optional
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
import torch
from PIL import ImageOps, Image


class QwenVLForEdit(BaseVLLMForEdit):

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        *args,
        **kwargs,
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
        }

        if device == "auto":
            load_kwargs["device_map"] = "auto"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model = self.model.eval().requires_grad_(False)

        self.device = self._infer_device()
        self._last_img_token_n = 0
        self._last_vt_range = None

        super().__init__(self.model, self.device, False)

    def get_img_token_n(self):
        """
        Return the number of image/visual tokens.

        Qwen2-VL uses dynamic visual token numbers depending on image size.
        Therefore, we return the value recorded during get_llm_input_embeds().
        """
        # return int(getattr(self, "_last_img_token_n", 0))
        return 576

    def is_q_former_based(self):
        """
        Qwen2-VL is not BLIP2/InstructBLIP Q-Former based.
        """
        return False

    def _infer_device(self):
        """Infer the actual device of the loaded model."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_base_model(self):
        """Get the inner base model for compatibility across transformers versions."""
        return getattr(self.model, "model", self.model)

    def _get_language_model(self):
        """Locate the language model module."""
        base_model = self._get_base_model()

        if hasattr(base_model, "language_model"):
            return base_model.language_model

        if hasattr(self.model, "language_model"):
            return self.model.language_model

        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model

        raise AttributeError("Cannot locate Qwen-VL language model module.")

    def _get_visual_model(self):
        """Locate the visual encoder module."""
        base_model = self._get_base_model()

        if hasattr(base_model, "visual"):
            return base_model.visual

        if hasattr(self.model, "visual"):
            return self.model.visual

        raise AttributeError("Cannot locate Qwen-VL visual module.")

    def _get_input_embeddings(self):
        """Locate the token embedding layer."""
        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb

        lang_model = self._get_language_model()

        if hasattr(lang_model, "embed_tokens"):
            return lang_model.embed_tokens

        raise AttributeError("Cannot locate token embedding layer.")

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def _build_qwen_vl_texts(self, texts: List[str], imgs: List[ImageClass]):
        """
        Build multimodal prompts for Qwen2-VL.
    
        Important:
        - If the input text already contains <|image_pad|>, do NOT apply chat_template again.
        - Otherwise apply Qwen2-VL chat_template to insert exactly one image token.
        """
        prompts = []
        image_token = self.get_img_special_token_str()
    
        for text, img in zip(texts, imgs):
            if image_token is not None and image_token in text:
                # Keep only one image token to match one image.
                if text.count(image_token) > 1:
                    first = text.find(image_token)
                    text = text[:first + len(image_token)] + text[first + len(image_token):].replace(image_token, "")
    
                prompts.append(text)
                continue
    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text},
                    ],
                }
            ]
    
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    
            # Safety: ensure only one image token for one image.
            if image_token is not None and prompt.count(image_token) > 1:
                first = prompt.find(image_token)
                prompt = prompt[:first + len(image_token)] + prompt[first + len(image_token):].replace(image_token, "")
    
            prompts.append(prompt)

        return prompts

    def _encode_images(self, pixel_values, image_grid_thw):
        """
        Encode images into visual token embeddings.

        Newer transformers versions may provide get_image_features().
        Older versions may require calling the visual module directly.
        """
        if hasattr(self.model, "get_image_features"):
            image_outputs = self.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            visual = self._get_visual_model()
            image_outputs = visual(
                pixel_values,
                grid_thw=image_grid_thw,
            )

        if hasattr(image_outputs, "pooler_output"):
            image_embeds = image_outputs.pooler_output
        elif isinstance(image_outputs, (tuple, list)):
            image_embeds = image_outputs[0]
        else:
            image_embeds = image_outputs

        return image_embeds

    def _get_position_ids(self, input_ids, attention_mask, image_grid_thw=None):
        """
        Try to reuse Qwen2-VL's native M-RoPE position_ids.

        If the current transformers version does not support this interface,
        return None and let the language model handle position ids internally.
        """
        base_model = self._get_base_model()

        if not hasattr(base_model, "get_rope_index"):
            return None

        try:
            image_token_id = self.get_img_special_token_id()
            video_token_id = getattr(self.model.config, "video_token_id", None)

            mm_token_type_ids = torch.zeros_like(input_ids)

            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1

            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2

            try:
                position_ids, _ = base_model.get_rope_index(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
            except TypeError:
                # Fallback for older transformers versions.
                position_ids, _ = base_model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    attention_mask,
                )

            return position_ids

        except Exception:
            return None

    def get_llm_input_embeds(
        self,
        texts: List[str],
        imgs: Optional[List[ImageClass]] = None,
    ):
        """
        Build language model inputs.

        Returns:
        - llm_inpt: A dictionary containing inputs_embeds, attention_mask, and position_ids.
        - vt_range: The start and end indices of visual tokens in the sequence.
        """
        embed_layer = self._get_input_embeddings()
        embed_device = embed_layer.weight.device

        if imgs is not None:
            if len(imgs) != len(texts):
                raise ValueError(
                    f"Expected len(imgs) == len(texts), "
                    f"but got {len(imgs)} images and {len(texts)} texts."
                )

            imgs = [
                ImageOps.pad(
                    img.convert("RGB"),
                    (672, 672),
                    method=Image.BICUBIC,
                    color=(0, 0, 0),
                )
                for img in imgs
            ]

            prompts = self._build_qwen_vl_texts(texts, imgs)

            inpt = self.processor(
                text=prompts,
                images=imgs,
                return_tensors="pt",
                padding=True,
            )

            image_grid_thw = inpt["image_grid_thw"]
            
            input_ids = inpt["input_ids"].to(embed_device)
            attention_mask = inpt["attention_mask"].to(embed_device)

            # Convert token ids into text embeddings.
            inputs_embeds = embed_layer(input_ids)

            pixel_values = inpt["pixel_values"]
            image_grid_thw = inpt["image_grid_thw"]

            visual = self._get_visual_model()
            visual_device = next(visual.parameters()).device

            if hasattr(visual, "get_dtype"):
                visual_dtype = visual.get_dtype()
            else:
                visual_dtype = next(visual.parameters()).dtype

            pixel_values = pixel_values.to(
                device=visual_device,
                dtype=visual_dtype,
            )
            image_grid_thw = image_grid_thw.to(visual_device)

            # Encode image patches into visual embeddings.
            image_embeds = self._encode_images(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            image_embeds = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

            # Ensure visual embeddings are shaped as [num_image_tokens, hidden_size].
            if image_embeds.dim() == 3:
                image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

            image_token_id = self.get_img_special_token_id()
            image_token_mask = input_ids == image_token_id

            n_image_tokens = int(image_token_mask.sum().item())
            n_image_features = int(image_embeds.shape[0])

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: "
                    f"tokens={n_image_tokens}, features={n_image_features}."
                )

            # Replace image placeholder token embeddings with visual embeddings.
            image_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds,
            )

            position_ids = self._get_position_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw.to(embed_device),
            )

            # Record the visual token range of the first sample in the batch.
            img_pos = torch.where(image_token_mask[0])[0]

            if len(img_pos) > 0:
                vt_range = [int(img_pos[0].item()), int(img_pos[-1].item()) + 1]
                self._last_img_token_n = vt_range[1] - vt_range[0]
                self._last_vt_range = vt_range
            else:
                vt_range = None
                self._last_img_token_n = 0
                self._last_vt_range = None

            llm_inpt = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
            }

            return llm_inpt, vt_range

        else:
            inpt = self.get_llm_tokenizer()(
                texts,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            input_ids = inpt.input_ids.to(embed_device)
            attention_mask = inpt.attention_mask.to(embed_device)
            inputs_embeds = embed_layer(input_ids)

            llm_inpt = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "position_ids": None,
            }

            return llm_inpt, None

    def get_llm_outpt(self, llm_inpt, vt_range=None):
        """Forward inputs through the full Qwen2-VL causal LM to get logits."""
    
        kwargs = {
            "inputs_embeds": llm_inpt["inputs_embeds"],
            "attention_mask": llm_inpt["attention_mask"],
            "output_attentions": None,
            "output_hidden_states": None,
            "return_dict": True,
            "use_cache": False,
        }
    
        if llm_inpt.get("position_ids", None) is not None:
            kwargs["position_ids"] = llm_inpt["position_ids"]
    
        return self.model(**kwargs)

    def get_img_special_token_str(self):
        """Return the Qwen2-VL image placeholder token."""
        return "<|image_pad|>"

    def get_img_special_token_id(self):
        """Return the image placeholder token id."""
        if hasattr(self.model.config, "image_token_id"):
            return self.model.config.image_token_id

        token = self.get_img_special_token_str()
        return self.get_llm_tokenizer().convert_tokens_to_ids(token)