import json
import os
import random
from typing import Dict, List, Optional
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from torchvision import datasets
from helpers.qwen_helpers import is_qwen_vl

class FlatImageDataset(Dataset):
    """Load images directly from a directory without class subfolders."""

    def __init__(self, root, transform=None):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.paths = sorted(os.path.join(root, name) for name in os.listdir(root)
                            if name.lower().endswith(extensions))
        if not self.paths:
            raise FileNotFoundError(f'No images found directly in {root}')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return (self.transform(image) if self.transform else image), 0


def _raw_image_dataset(root):
    has_class_dir = any(entry.is_dir() for entry in os.scandir(root))
    return (datasets.ImageFolder(root=root, transform=None) if has_class_dir
            else FlatImageDataset(root, transform=None))


def _limit_direction_dataset(dataset, args):
    limit = getattr(args, "direction_samples", None)
    return (Subset(dataset, range(min(limit, len(dataset))))
            if limit is not None else dataset)


def get_val_loader(args, dataset_dir, image_processor=None, batch_size=32, workers=8):
    if is_qwen_vl(args.model_name):
        dataset = _limit_direction_dataset(_raw_image_dataset(dataset_dir), args)
        def qwen_collate(batch):
            images = [image.convert("RGB") for image, _ in batch]
            encoded = image_processor(images=images, return_tensors="pt")
            return {"pixel_values": encoded["pixel_values"],
                    "image_grid_thw": encoded["image_grid_thw"]}
        return DataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=workers, pin_memory=True,
                          collate_fn=qwen_collate)

    if args.model_name == "blip2":
        dataset = _limit_direction_dataset(_raw_image_dataset(dataset_dir), args)
        def blip_collate(batch):
            images = [image.convert("RGB") for image, _ in batch]
            return image_processor(images=images, return_tensors="pt")["pixel_values"]
        return DataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=workers, pin_memory=True,
                          collate_fn=blip_collate)

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    has_class_dir = any(entry.is_dir() for entry in os.scandir(dataset_dir))
    dataset = (datasets.ImageFolder(root=dataset_dir, transform=val_transform)
               if has_class_dir else FlatImageDataset(dataset_dir, val_transform))
    dataset = _limit_direction_dataset(dataset, args)
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=16, pin_memory=True)


# def collate_fn(config, batch, image_processor):

#     images, labels = zip(*batch)
    
#     processed = image_processor(images=list(images), return_tensors="pt")
#     if config.model_name in ['llava', 'blip2']:
#         image_values = processed["pixel_values"]
#     elif is_qwen_vl(config.model_name):
#         image_values = {
#             "pixel_values": processed["pixel_values"],
#             "grid_thw": processed["image_grid_thw"]
#         }

#     return image_values, labels


# def get_val_loader(args, image_processor, dataset_dir):

#     dataset = datasets.ImageFolder(
#         root=dataset_dir,
#         transform=None
#     )

#     val_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         # num_workers=4,
#         pin_memory=True,
#         num_workers=16,
#         persistent_workers=True,
#         prefetch_factor=4,
#         collate_fn=lambda batch: collate_fn(args, batch, image_processor)
#     )

#     return val_loader

class AKEVLLMEditData:
    
    def __init__(self, image_processor, config):

        self.model_name = config.model_name
        self.img_root = config.img_root
        self.img_root_modified = config.img_root_modified
        self.img_masks = config.img_masks
        self.image_processor = image_processor

        if config.model_name == "llava":
            self.prompt = "USER: <image>\n {} Please answer in one word. ASSISTANT:"
        elif config.model_name == "blip2":
            self.prompt = "Question: {} Please answer in one word. Short answer:"
        elif is_qwen_vl(config.model_name):
            self.prompt = "{} Please answer in one word."


    def _get_img_masks(self, config, processor, image, image_modified, masks, type_modified, text_prompt=None, threshold=128):

        image = Image.open(image).convert("RGB")
        image_modified = Image.open(image_modified).convert("RGB")
        masks_img = Image.open(masks)
        
        masks_array = np.array(masks_img.convert("L"))
        masks_binary = (masks_array > threshold).astype(np.float32)
        masks_tensor = torch.from_numpy(masks_binary).unsqueeze(0)  # [1, H, W]

        if config.model_name in ["llava", "blip2"]:
            img_tensor = processor(images=image, return_tensors="pt").pixel_values[0]
            modified_tensor = processor(images=image_modified, return_tensors="pt").pixel_values[0]
        elif is_qwen_vl(config.model_name):
            img_tensor = dict(processor(images=image, return_tensors="pt"))
            modified_tensor = dict(processor(images=image_modified, return_tensors="pt"))

        return img_tensor, modified_tensor, masks_tensor


    def _add_data_with_img(self, target_list, image: Optional[str], prompt: Optional[str], target: Optional[str]):
        if image is None or prompt is None or target is None:
            return
        target_list.append({'image': image, 'prompt': self.prompt.format(prompt), 'target': target})


    def _join_img(self, img_root: str, name: Optional[str]) -> Optional[str]:
        if name is None or name == '':
            return None
        return os.path.join(img_root, name)
    
    def answer_mapping(self, answer):
        llava_mapping = {
            "cube": "Square",
            "sphere": "Round",
            "cylinder": "Cylinder"
        }

        if self.model_name == 'llava':
            return llava_mapping[answer] if answer in llava_mapping.keys() else answer.capitalize()
        else:
            return answer

    def get_dataset(self, config) -> List[Dict]:
        
        items: List[Dict] = []

        with open(config.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        import random
        # dataset = random.sample(dataset, 50)
        for d in tqdm(dataset):
            
            # if d['attribute_type'] == 'color':
            #     continue

            folder = d.get('image').split('/')[0]
            raw_name = d.get('image')
            original_direct = self._join_img(self.img_root, raw_name)
            edited_pair = f"_{d.get('pred')}_{d.get('alt')}_"
            original_name = raw_name.replace(edited_pair, "_", 1)
            image_modified = (original_direct if os.path.exists(original_direct)
                              else self._join_img(self.img_root, original_name))
            direct_image = self._join_img(self.img_root_modified, raw_name)
            direct_mask = self._join_img(self.img_masks, d.get('image'))
            legacy_name = d.get('image').replace(
                folder + "_",
                "_".join([folder, d.get('pred'), d.get('alt')]) + "_")
            image = (direct_image if os.path.exists(direct_image)
                     else self._join_img(self.img_root_modified, legacy_name))
            masks = (direct_mask if os.path.exists(direct_mask)
                     else self._join_img(self.img_masks, legacy_name))
            img_tensor, modified_imgs_tensor, masks = self._get_img_masks(config, self.image_processor, image, image_modified, masks, d.get('attribute_type'))

            gen_img = self._join_img(self.img_root, d.get('gen_img'))
            # AttributeDataset calls these multimodal-locality (m_loc) and
            # text-locality (t_loc). They are the in/out locality probes and
            # share the designated locality image.
            loc_image = self._join_img(self.img_root, d.get('m_loc_image') or d.get('loc_img'))
            
            ################
            # Reliability
            ################
            rel = []
            self._add_data_with_img(rel, image_modified, d.get('src'), self.answer_mapping(d.get('alt')))

            ################
            # Locality
            ################
            loc = {'loc_in': [], 'loc_out': []}
            self._add_data_with_img(loc['loc_in'], loc_image, d.get('m_loc_q') or d.get('loc_in_q'), "_")
            self._add_data_with_img(loc['loc_out'], loc_image, d.get('t_loc_q') or d.get('loc_out_q'), "_")

            ################
            # generality
            ################
            gen = {k: [] for k in ['rephrase_image', 'gen1', 'gen2']}
            rimg = self._join_img(self.img_root, d.get('rephrase_image'))
            self._add_data_with_img(gen['rephrase_image'], rimg, d.get('src'), self.answer_mapping(d.get('alt')))
            self._add_data_with_img(gen['gen1'], image_modified, d.get('gen1_q'), self.answer_mapping(d.get('gen1_a')))
            self._add_data_with_img(gen['gen2'], gen_img, d.get('gen2_q_1'), self.answer_mapping(d.get('gen2_a')))
            self._add_data_with_img(gen['gen2'], gen_img, d.get('gen2_q_2'), self.answer_mapping(d.get('gen2_a')))
                
            item = {
                "inner": {
                    'image': image_modified,
                    'modified_image': image,
                    # Learn the edit in the intended direction: the original
                    # image is the key and the visually edited image supplies
                    # the target representation.
                    'imgs': modified_imgs_tensor,
                    'modified_imgs': img_tensor,
                    'masks': masks,
                    'prompt': d.get('src'),
                    'target': self.answer_mapping(d.get('alt')),
                },
                    "rel": rel,
                    "loc": loc,
                    "gen": gen,
                    "locality_candidate": {
                        "image": image_modified, "prompt": d.get("src"),
                        "target": self.answer_mapping(d.get("pred")),
                        "attribute_type": d.get("attribute_type")},
                    "metadata": {
                        "attribute_type": d.get("attribute_type"),
                        "related_images": [p for p in
                            (image_modified, image, rimg, gen_img) if p]}
            }
            items.append(item)
        
        return items
