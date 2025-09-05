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
                image = os.path.join(self.vis_root, record["image"].split('_')[1], record["image"])
                rephrase_image = [os.path.join(self.vis_root, v.split('_')[1],v) for k, v in record.items() if "rephrase_image" in k]
                generality_image = [os.path.join(self.vis_root, v.split('_')[1], v) for k, v in record.items() if "gen_img" in k and "q" not in k and "a" not in k]
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
            item['rephrase_image_ground_truth'] = [" " + record['alt']] * len(rephrase_image)

            ############################################################

            #locality
            item['locality_prompt_1'] = [v for k, v in record.items() if "loc_q_1" in k]
            item['locality_ground_truth_1'] = [" " + v for k, v in record.items() if "loc_a_1" in k]
            item['img_locality_1'] = [item['image']] * len(item['locality_prompt_1'])

            item['locality_prompt_2'] = record['loc_q_2']
            item['locality_ground_truth_2'] = " " + record['loc_a_2']
            item['img_locality_2'] = item['image']

            ############################################################

            #generality
            item['generality_prompt_1'] = record['gen_q_1']
            item['generality_ground_truth_1'] = " " + record['gen_a_1']
            item['img_generality_1'] = image

            item['generality_prompt_2'] = [v for k, v in record.items() if "gen_q_2" in k or "gen_q_3" in k]
            item['generality_ground_truth_2'] = [" " + v for k, v in record.items() if "gen_a_2" in k or "gen_a_3" in k]
            item['img_generality_2'] = [image]*len(item['generality_prompt_2'])

            item['generality_prompt_3'] = record['gen_q_4']
            item['generality_ground_truth_3'] = " " + record['gen_a_4']
            item['img_generality_3'] = image

            ############################################################

            #2-img generality
            item['generality_image_1'] = generality_image
            item['generality_image_prompt_1'] = [v for k, v in record.items() if "gen_img" in k and "q_1" in k]
            item['generality_image_ground_truth_1'] = [" " + v for k, v in record.items() if "gen_img" in k and "a_1" in k]

            item['generality_image_2'] = [generality_image[0]]*2 + [generality_image[1]]*2
            item['generality_image_prompt_2'] = [v for k, v in record.items() if "gen_img" in k and "q_2" in k]
            item['generality_image_ground_truth_2'] = [" " + v for k, v in record.items() if "gen_img" in k and "a_2" in k]

            item['generality_image_3'] = generality_image
            item['generality_image_prompt_3'] = [v for k, v in record.items() if "gen_img" in k and "q_3" in k and "pre" not in k]
            item['generality_image_ground_truth_3'] = [" " + v for k, v in record.items() if "gen_img" in k and "a_3" in k and "pre" not in k]

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

        ############################################################

        #locality
        img_loc_1 = [b["img_locality_1"] for b in batch]
        loc_q_1 = [b["locality_prompt_1"] for b in batch]
        loc_a_1 = [b['locality_ground_truth_1'] for b in batch]

        img_loc_2 = [b["img_locality_2"] for b in batch]
        loc_q_2 = [b["locality_prompt_2"] for b in batch]
        loc_a_2 = [b['locality_ground_truth_2'] for b in batch]

        ############################################################
        
        #generality
        img_gen_1 = [b["img_generality_1"] for b in batch]
        gen_q_1 = [b['generality_prompt_1'] for b in batch]
        gen_a_1 = [b['generality_ground_truth_1'] for b in batch]

        img_gen_2 = [b["img_generality_2"] for b in batch]
        gen_q_2 = [b['generality_prompt_2'] for b in batch]
        gen_a_2 = [b['generality_ground_truth_2'] for b in batch]

        img_gen_3 = [b["img_generality_3"] for b in batch]
        gen_q_3 = [b['generality_prompt_3'] for b in batch]
        gen_a_3 = [b['generality_ground_truth_3'] for b in batch]

        ############################################################
        
        #generality_image
        gen_images_1 = [b['generality_image_1'] for b in batch]
        gen_img_q_1 = [b['generality_image_prompt_1'] for b in batch]
        gen_img_a_1 = [b['generality_image_ground_truth_1'] for b in batch]

        gen_images_2 = [b['generality_image_2'] for b in batch]
        gen_img_q_2 = [b['generality_image_prompt_2'] for b in batch]
        gen_img_a_2 = [b['generality_image_ground_truth_2'] for b in batch]

        gen_images_3 = [b['generality_image_3'] for b in batch]
        gen_img_q_3 = [b['generality_image_prompt_3'] for b in batch]
        gen_img_a_3 = [b['generality_image_ground_truth_3'] for b in batch]

        ############################################################

        # edit_inner
        edit_inner = {}
        edit_inner['type'] = attribute_type
        edit_inner['image'] = image
        edit_inner['cond'] = cond
        edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        ############################################################
            
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

        ############################################################
            
        # loc
        loc_1 = {}
        loc_1['image'] = img_loc_1
        loc_1['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(loc_q_1, loc_a_1)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_1['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in loc_q_1]
            loc_1['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in loc_a_1]
        else:
            loc_1['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in loc_q_1]
            loc_1['labels'] = [self.tok(answer, return_tensors="pt",)["input_ids"] for answer in loc_a_1]

        loc_2 = {}
        loc_2['image'] = img_loc_2
        loc_2['text_input'] = [self.prompt.format(q) + a for q, a in zip(loc_q_2, loc_a_2)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_2['prompts_len'] = [len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in loc_q_2]
            loc_2['labels'] = self.tok(loc_a_2, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_2['prompts_len'] = [len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in loc_q_2]
            loc_2['labels'] = self.tok(loc_a_2, return_tensors="pt",)["input_ids"]

        ############################################################
            
        # gen
        gen_1 = {}
        gen_1['image'] = img_gen_1
        gen_1['text_input'] = [self.prompt.format(q) + a for q, a in zip(gen_q_1, gen_a_1)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_1['prompts_len'] = [len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in gen_q_1]
            gen_1['labels'] = self.tok(gen_a_1, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            gen_1['prompts_len'] = [len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in gen_q_1]
            gen_1['labels'] = self.tok(gen_a_1, return_tensors="pt",padding=True)["input_ids"]

        gen_2 = {}
        gen_2['image'] = img_gen_2
        gen_2['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_q_2, gen_a_2)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_2['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_q_2]
            gen_2['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_a_2]
        else:
            gen_2['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_q_2]
            gen_2['labels'] = [self.tok(answer, return_tensors="pt",padding=True)["input_ids"] for answer in gen_a_2]
        
        gen_3 = {}
        gen_3['image'] = img_gen_3
        gen_3['text_input'] = [self.prompt.format(q) + a for q, a in zip(gen_q_3, gen_a_3)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_3['prompts_len'] = [len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in gen_q_3]
            gen_3['labels'] = self.tok(gen_a_3, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            gen_3['prompts_len'] = [len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in gen_q_3]
            gen_3['labels'] = self.tok(gen_a_3, return_tensors="pt",padding=True)["input_ids"]

        ############################################################
            
        # gen-img
        gen_img_1 = {}
        gen_img_1['image'] = gen_images_1
        gen_img_1['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_img_q_1, gen_img_a_1)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_img_1['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_img_q_1]
            gen_img_1['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_img_a_1]
        else:
            gen_img_1['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_img_q_1]
            gen_img_1['labels'] = [self.tok(answer, return_tensors="pt",)["input_ids"] for answer in gen_img_a_1]
        
        gen_img_2 = {}
        gen_img_2['image'] = gen_images_2
        gen_img_2['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_img_q_2, gen_img_a_2)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_img_2['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_img_q_2]
            gen_img_2['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_img_a_2]
        else:
            gen_img_2['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_img_q_2]
            gen_img_2['labels'] = [self.tok(answer, return_tensors="pt",padding=True)["input_ids"] for answer in gen_img_a_2]
        
        gen_img_3 = {}
        gen_img_3['image'] = gen_images_3
        gen_img_3['text_input'] = [[self.prompt.format(q[idx]) + a[idx] for idx in range(len(q))] for q, a in zip(gen_img_q_3, gen_img_a_3)]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            gen_img_3['prompts_len'] = [[len(self.tok.encode(self.prompt.format(tmp), add_special_tokens=False)) for tmp in q] for q in gen_img_q_3]
            gen_img_3['labels'] = [self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"] for answer in gen_img_a_3]
        else:
            gen_img_3['prompts_len'] = [[len(self.tok.encode(tmp, add_special_tokens=False)) for tmp in q] for q in gen_img_q_3]
            gen_img_3['labels'] = [self.tok(answer, return_tensors="pt",)["input_ids"] for answer in gen_img_a_3]

        ############################################################
            
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
            
            #locality
            "loc_1": loc_1,
            "loc_2": loc_2,
            #generality
            "gen_1": gen_1,
            "gen_2": gen_2,
            "gen_3": gen_3,
            #geenrality_image
            "gen_img_1": gen_img_1,
            "gen_img_2": gen_img_2,
            "gen_img_3": gen_img_3,

            "cond": cond
        }
        return dict_to(batch, self.config.device)
