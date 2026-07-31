from typing import Dict, List, Optional
from copy import deepcopy
import json, os
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from .vllm import BaseVLLMEditData
from utils.GLOBAL import model_path_map

class AKEVLLMEditData(BaseVLLMEditData):
    
    def __init__(self, ake_json_path: str, img_root_dir: str, loc_img_root_dir:str, model_name: str, data_n: Optional[int] = None,
                 add_leading_space_for_targets: bool = False) -> None:

        self.model_name = model_name
        self.processor = None
        self.add_leading_space_for_targets = add_leading_space_for_targets
        data_with_img_path = self._build_from_ake(ake_json_path, img_root_dir, loc_img_root_dir, data_n)
        data_with_img = deepcopy(data_with_img_path)
            
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'AKE'

    def _join_img(self, img_root_dir: str, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        if name == '':
            return None
        return os.path.join(img_root_dir, name)

    def qwenvl_qa(self, image, question):
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1280*28*28,},
                    {"type": "text", "text": self.prompt.format(question)}
                ]
            }]
    
    def _add_if_all(self, target_list: List[Dict], image: Optional[str], prompt: Optional[str], target: Optional[str]):

        if prompt is None or target is None:
            return
            
        if self.model_name == "blip2":
            target_list.append({'image': image, 'prompt': self.prompt.format(prompt), 'target': ' ' + target.capitalize()})
        elif self.model_name == "llava":
            target_list.append({'image': image, 'prompt': self.prompt.format(prompt), 'target': target.capitalize()})
        elif self.model_name == "qwenvl":
            target_list.append({'image': image,
                                'prompt': self.processor.apply_chat_template(self.qwenvl_qa(image, prompt), tokenize=False, add_generation_prompt=True), 
                                'target': target})

            
    def _build_from_ake(self, ake_json_path: str, img_root_dir: str, loc_img_root_dir: str, data_n: Optional[int]) -> List[Dict]:

        print(self.model_name)
        
        if self.model_name == "blip2":
            self.prompt = "Question: {} Please answer in one word. Short answer:"
        elif self.model_name == "llava":
            self.prompt = "USER: <image> \n {} Please answer in one word. ASSISTANT:"
        elif self.model_name == "qwenvl":
            self.prompt = "{} Please answer in one word."
            self.processor = AutoProcessor.from_pretrained(model_path_map[self.model_name], trust_remote_code=True)
        
        with open(ake_json_path, 'r') as f:
            raw = json.load(f)
        if data_n is None:
            data_n = len(raw)
        data_n = min(data_n, len(raw))

        items: List[Dict] = []
        for i in tqdm(range(data_n), 'Building AKE data'):
            d = raw[i]

            request_img = self._join_img(img_root_dir, d.get('image'))
            if self.model_name == "blip2":
                request_target = ' ' + d.get('alt').capitalize()
            elif self.model_name == "qwenvl":
                request_target = d.get('alt')
            elif self.model_name == "llava":
                request_target = d.get('alt').capitalize()

            if self.model_name == "qwenvl":
                request_prompt = self.processor.apply_chat_template(self.qwenvl_qa(request_img, d.get('src')), tokenize=False, add_generation_prompt=True)
            else:
                request_prompt = self.prompt.format(d.get('src'))

            
            if request_prompt is None or request_target is None or request_img is None:
                # skip malformed entries
                continue

            # image
            rimg = self._join_img(img_root_dir, d.get('rephrase_image'))
            gen_img = self._join_img(img_root_dir, d.get('gen_img'))
            loc_img = self._join_img(img_root_dir, d.get('loc_img'))
            
            # prepare generality buckets
            gen = {'rephrase_image':[], 'gen1':[], 'gen2':[]}
            self._add_if_all(gen['rephrase_image'], rimg, d.get('src'), d.get('alt'))
            self._add_if_all(gen['gen1'], gen_img, d.get('gen1_q'), d.get('gen1_a')) 
            self._add_if_all(gen['gen2'], gen_img, d.get('gen2_q_1'), d.get('gen2_a')) 
            self._add_if_all(gen['gen2'], gen_img, d.get('gen2_q_2'), d.get('gen2_a')) 

            
            # Locality
            loc = {'Loc_in':[], 'Loc_out':[]}
            self._add_if_all(loc['Loc_in'], loc_img, d.get('loc_in_q'), "_")
            self._add_if_all(loc['Loc_out'], loc_img, d.get('loc_out_q'), "_")

            
            item = {
                'request': {
                    'image': request_img,
                    'prompt': request_prompt,
                    'target_new': request_target,
                },
                'generality': gen,
                'locality': loc
            }

            items.append(item)
        return items
