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
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def get_val_loader(dataset_dir, batch_size=32, workers=8):

    val_transform = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop(336),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225])   
    ])

    dataset = datasets.ImageFolder(root=dataset_dir, transform=val_transform)

    # num_samples = 1000
    # all_indices = list(range(len(dataset)))
    # subset_indices = random.sample(all_indices, min(num_samples, len(dataset)))
    # subset_dataset = Subset(dataset, subset_indices)

    val_loader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return val_loader

class AKEVLLMEditData:
    
    def __init__(self, image_processor, config):

        self.img_root = config.img_root
        self.img_root_modified = config.img_root_modified
        self.img_masks = config.img_masks
        self.image_processor = image_processor

        if config.model_name == "llava":
            self.prompt = "USER: <image>\n {} Please answer in one word. ASSISTANT:"
        elif config.model_name == "blip2":
            self.prompt = "Question: {} Please answer in one word. Short answer:"
        elif config.model_class == "qwen-vl":
            self.prompt = "{} Please answer in one word."


    def _get_img_masks(self, processor, image, image_modified, masks, type_modified, text_prompt=None, threshold=128):

        image = Image.open(image).convert("RGB")
        image_modified = Image.open(image_modified).convert("RGB")
        masks_img = Image.open(masks)
        
        masks_array = np.array(masks_img.convert("L"))
        masks_binary = (masks_array > threshold).astype(np.float32)
        masks_tensor = torch.from_numpy(masks_binary).unsqueeze(0)  # [1, H, W]

        img_tensor = processor(images=image, return_tensors="pt").pixel_values[0]
        modified_tensor = processor(images=image_modified, return_tensors="pt").pixel_values[0]

        return img_tensor, modified_tensor, masks_tensor


    def _add_data_with_img(self, target_list, image: Optional[str], prompt: Optional[str], target: Optional[str]):
        if prompt is not None or target is not None:
            target_list.append({'image': image, 'prompt': self.prompt.format(prompt), 'target': target})


    def _join_img(self, img_root: str, name: Optional[str]) -> Optional[str]:
        if name is None or name == '':
            return None
        return os.path.join(img_root, name)
    

    def get_dataset(self, config) -> List[Dict]:
        
        items: List[Dict] = []

        with open(config.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        for d in tqdm(dataset):

            image_modified = self._join_img(self.img_root, d.get('image'))
            image = self._join_img(self.img_root_modified, d.get('image').replace(".png", "_" + d.get('alt') + ".png"))
            masks = self._join_img(self.img_masks, d.get('image').replace(".png", "_" + d.get('alt') + ".png"))
            img_tensor, modified_imgs_tensor, masks = self._get_img_masks(self.image_processor, image, image_modified, masks, d.get('attribute_type'))

            gen_img = self._join_img(self.img_root, d.get('gen_img'))
            loc_m_image = self._join_img(self.img_root, d.get('m_loc_image'))
            loc_t_image = self._join_img(self.img_root, config.t_loc_image)
            
            ################
            # Reliability
            ################
            rel = []
            self._add_data_with_img(rel, image_modified, d.get('src'), d.get('alt'))

            ################
            # Locality
            ################
            loc = {'Loc_m': [], 'Loc_t': []}
            loc_m_q, loc_m_a = d.get('loc_m_q'), "_"
            self._add_data_with_img(loc['Loc_m'], loc_m_image, loc_m_q, loc_m_a)
            loc_t_q, loc_t_a = d.get('loc_t_q'), "_"
            self._add_data_with_img(loc['Loc_t'], loc_t_image, loc_t_q, loc_t_a)

            ################
            # generality
            ################
            gen = {k: [] for k in ['re_img', 'gen1', 'gen2']}
            rimg = self._join_img(self.img_root, d.get('rephrase_image'))
            self._add_data_with_img(gen['re_img'], rimg, d.get('src'), d.get('alt'))
            self._add_data_with_img(gen['gen1'], image_modified, d.get('gen1_q'), d.get('gen1_a'))
            self._add_data_with_img(gen['gen2'], gen_img, d.get('gen2_q_1'), d.get('gen2_a'))
            self._add_data_with_img(gen['gen2'], gen_img, d.get('gen2_q_2'), d.get('gen2_a'))
                
            item = {
                "inner": {
                    'imgs': img_tensor,
                    'modified_imgs': modified_imgs_tensor,
                    'masks': masks,
                    'prompt': d.get('src'),
                    'target': d.get('alt'),
                },
                    "rel": rel,
                    "loc": loc,
                    "gen": gen
            }
            items.append(item)
        
        return items
