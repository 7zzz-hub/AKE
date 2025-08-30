"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from tqdm import tqdm

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token, process_images
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import CLIPProcessor

class AttributeDataset(BaseDataset):
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class in ["Blip2OPT", "minigpt4"]:
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.image
        super().__init__(vis_processor, vis_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Please answer in one word. Short answer:"

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]  
        for i, record in tqdm(enumerate(self.annotation),total=len(self.annotation),desc="Loading AttributeDataset"):
            
            if record['alt'] == "":
                continue
            
            if self.config.model_class in ["LLaVA", "Blip2OPT","qwen-vl","minigpt4"]:
                image = os.path.join(self.vis_root, record["image"])
                rephrase_image = [os.path.join(self.vis_root,v) for k, v in record.items() if "rephrase_image" in k]
                generality_image = [os.path.join(self.vis_root,v) for k, v in record.items() if "gen_img" in k and "q" not in k and "a" not in k]
            elif self.config.model_name == "owl-2":  
                image_path = os.path.join(self.vis_root, record["image"])
                _image = Image.open(image_path).convert('RGB')
                max_edge = max(_image.size) 
                image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)
                #generality
                gen_img_path = [os.path.join(self.vis_root,v) for k, v in record.items() if "gen_img" in k and 'q' not in k and 'a' not in k]
                generality_image = [Image.open(img_path).convert("RGB") for img_path in gen_img_path]
                generality_image = [process_images([img.resize((max_edge, max_edge))], self.vis_processor) for img in generality_image]
            else:
                raise NotImplementedError
      
            item = {
                'attribute_type': record['attribute_type'],
                'prompt': record['src'],
                'pred': record['pred'],
                'target': " " + record['alt'],
                'image': image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }

            #3 rephrase images
            item['rephrase_image'] = rephrase_image
            item['rephrase_image_prompt'] = [record['src']]*len(rephrase_image)
            item['rephrase_image_ground_truth'] = [" " + record['alt']]*len(rephrase_image)
            #2-img generality
            item['generality_image'] = [generality_image[0]]*3 + [generality_image[1]]*3
            item['generality_image_prompt'] = [v for k, v in record.items() if "gen_img" in k and "q" in k]
            item['generality_image_ground_truth'] = [" " + str(v) for k, v in record.items() if "gen_img" in k and "a" in k and "pre" not in k]
            #locality
            item['locality_prompt'] = [record['loc_q_1'], record['loc_q_2']]
            item['locality_ground_truth'] = [" " + record['loc_a_1'], " " + record['loc_a_2']]
            item['img_locality'] = [image]*len(item['locality_prompt'])
            #generality
            item['generality_prompt'] = [v for k, v in record.items() if "gen_q" in k]
            item['generality_ground_truth'] = [" " + str(v) for k, v in record.items() if "gen_a" in k and "pre" not in k]
            item['img_generality'] = [image]*len(item['generality_prompt'])
            data.append(item)
             
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        attribute_type = [b['attribute_type'] for b in batch]
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        image = [b['image'] for b in batch]
        #rephrase image
        rephrase_image = [b["rephrase_image"] for b in batch]
        rephrase_image_q = [b["rephrase_image_prompt"] for b in batch]
        rephrase_image_a = [b['rephrase_image_ground_truth'] for b in batch]
        #locality
        img_loc = [b["img_locality"] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b['locality_ground_truth'] for b in batch]
        #generality
        img_gen = [b["img_generality"] for b in batch]
        gen_q = [b['generality_prompt'] for b in batch]
        gen_a = [b['generality_ground_truth'] for b in batch]
        #generality_image
        gen_images = [b['generality_image'] for b in batch]
        gen_img_q = [b['generality_image_prompt'] for b in batch]
        gen_img_a = [b['generality_image_ground_truth'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['type'] = attribute_type
        edit_inner['image'] = image
        edit_inner['cond'] = cond
        # edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        

        #rephrase image
        re_image = {}
        re_image['image'] = rephrase_image
        re_image['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(rephrase_image_q, rephrase_image_a)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            re_image['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in rephrase_image_q]
            re_image['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in rephrase_image_a]
        else:
            re_image['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in rephrase_image_q]
            re_image['labels'] = [self.tok(answer, return_tensors="pt",padding=True)["input_ids"] for answer in rephrase_image_a]

        # loc
        loc = {}
        loc['image'] = img_loc
        loc['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(loc_q, loc_a)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in loc_q]
            loc['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in loc_a]
        else:
            loc['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in loc_q]
            loc['labels'] = [self.tok(answer, return_tensors="pt",)["input_ids"] for answer in loc_a]

        # gen
        gen = {}
        gen['image'] = img_gen
        gen['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_q, gen_a)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_q]
            gen['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_a]
        else:
            gen['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_q]
            gen['labels'] = [self.tok(answer, return_tensors="pt",padding=True)["input_ids"] for answer in gen_a]

        # gen-img
        gen_img = {}
        gen_img['image'] = gen_images
        gen_img['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_img_q, gen_img_a)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_img['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_img_q]
            gen_img['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_img_a]
        else:
            gen_img['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_img_q]
            gen_img['labels'] = [self.tok(answer, return_tensors="pt",)["input_ids"] for answer in gen_img_a]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "re_image": re_image,
            "loc": loc,
            "gen": gen,
            "gen_img": gen_img,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
