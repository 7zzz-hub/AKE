from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from PIL import Image

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        # get tokenizer and vis_processor
        if config.model_class in ["Blip2OPT", "minigpt4"]:
            self.vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            self.vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            self.vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            self.vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")
        
        self.model_class = config.model_class

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        ################################ PRE-EDIT LOCALITY  ################################

        with torch.no_grad():

            #locality_q_1
            base_logits_1 = []
            for idx in range(len(batch["loc_1"]['labels'])):
                tmp = {}
                for k in batch["loc_1"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["loc_1"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["loc_1"][k][idx]
                base_outputs = self.model(tmp)
                if not isinstance(base_outputs, torch.Tensor):
                    base_logits_1.append(base_outputs.logits)
                else:  
                    base_logits_1.append(base_outputs)
            base_logits_1 = torch.stack(base_logits_1, dim=0)

            #gpu
            if isinstance(tmp, torch.Tensor):
                del tmp

            #locality_q_2
            if self.model_class == "LLaVA":
                image = [self.vis_processor(Image.open(img_path), return_tensors='pt')['pixel_values'] for img_path in batch["loc_2"]["image"]]
            else:
                image = [self.vis_processor(Image.open(img_path).convert("RGB")) for img_path in batch["loc_2"]["image"]]
            batch["loc_2"]["image"] = torch.stack(image, dim=0)
            base_outputs = self.model(batch["loc_2"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits_2 = base_outputs.logits
            else:  
                base_logits_2 = base_outputs
                
        ####################################################################################################

        # Do the edit
        start = time.time()

        if self.model_class == "LLaVA":
            image = [self.vis_processor(Image.open(img_path), return_tensors='pt')['pixel_values'] for img_path in batch["edit_inner"]["image"]]
        else:
            image = [self.vis_processor(Image.open(img_path).convert("RGB")) for img_path in batch["edit_inner"]["image"]]
        batch["edit_inner"]["image"] = torch.stack(image, dim=0)
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])

        edit_time = time.time() - start

        info_dict = {}
        l_total, l_image_edit, l_loc, l_base = 0, 0, 0, 0, 
        l_edit_1, l_edit_2, l_edit_3 = 0, 0, 0
        l_gen_img_edit_1, l_gen_img_edit_2, l_gen_img_edit_3 = 0, 0, 0
        
        with torch.set_grad_enabled(training):
            
            ################################ GENERALITY  ################################

            if self.model_class == "LLaVA":
                image = [self.vis_processor(Image.open(img_path), return_tensors='pt')['pixel_values'] for img_path in batch["gen_1"]["image"]]
            else:
                image = [self.vis_processor(Image.open(img_path).convert("RGB")) for img_path in batch["gen_1"]["image"]]
            batch["gen_1"]["image"] = torch.stack(image, dim=0)
            post_edit_outputs = edited_model(batch["gen_1"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits_1 = post_edit_outputs.logits
            else:  
                post_edit_logits_1 = post_edit_outputs
            post_batch_labels_1 = batch["gen_1"]["labels"]

            if post_edit_logits_1.shape[1] > post_batch_labels_1.shape[1]:
                l_edit_1 = self.model.edit_loss_fn(self.config, post_edit_logits_1, post_batch_labels_1)["nll"]
            else:
                l_edit_1 = self.model.edit_loss_fn(self.config, post_edit_logits_1, post_batch_labels_1[:, -post_edit_logits_1.shape[1]-1:])["nll"]

            ###########
            post_edit_logits_2 = []
            for idx in range(len(batch["gen_2"]['labels'])):
                tmp = {}
                for k in batch["gen_2"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["gen_2"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["gen_2"][k][idx]
                post_edit_outputs = edited_model(tmp)
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_edit_logits_2.append(post_edit_outputs.logits)
                else:  
                    post_edit_logits_2.append(post_edit_outputs)
            post_edit_logits_2 = torch.stack(post_edit_logits_2, dim=0)
            post_batch_labels_2 = batch["gen_2"]["labels"]

            for _post_edit_logits, _post_batch_labels in zip(post_edit_logits_2, post_batch_labels_2):
                if _post_edit_logits.shape[1] > _post_batch_labels.shape[1]:
                    l_edit_2 += self.model.edit_loss_fn(self.config, _post_edit_logits, _post_batch_labels)["nll"]
                else:
                    l_edit_2 += self.model.edit_loss_fn(self.config, _post_edit_logits, _post_batch_labels[:, -_post_edit_logits.shape[1]-1:])["nll"]
            l_edit_2 /= post_edit_logits_2.shape[0]

            ###########
            if self.model_class == "LLaVA":
                image = [self.vis_processor(Image.open(img_path), return_tensors='pt')['pixel_values'] for img_path in batch["gen_3"]["image"]]
            else:
                image = [self.vis_processor(Image.open(img_path).convert("RGB")) for img_path in batch["gen_3"]["image"]]
            batch["gen_3"]["image"] = torch.stack(image, dim=0)
            post_edit_outputs = edited_model(batch["gen_3"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits_3 = post_edit_outputs.logits
            else:  
                post_edit_logits_3 = post_edit_outputs
            post_batch_labels_3 = batch["gen_3"]["labels"]

            if post_edit_logits_3.shape[1] > post_batch_labels_3.shape[1]:
                l_edit_3 = self.model.edit_loss_fn(self.config, post_edit_logits_3, post_batch_labels_3)["nll"]
            else:
                l_edit_3 = self.model.edit_loss_fn(self.config, post_edit_logits_3, post_batch_labels_3[:, -post_edit_logits_3.shape[1]-1:])["nll"]


            ################################ GENERALITY IMAGE ################################

            post_gen_image_edit_logits_1 = []
            for idx in range(len(batch["gen_img_1"]['labels'])):
                tmp = {}
                for k in batch["gen_img_1"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["gen_img_1"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["gen_img_1"][k][idx]
                post_edit_outputs = edited_model(tmp)
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_gen_image_edit_logits_1.append(post_edit_outputs.logits)
                else:  
                    post_gen_image_edit_logits_1.append(post_edit_outputs)
            post_gen_image_edit_logits_1 = torch.stack(post_gen_image_edit_logits_1, dim=0)
            post_gen_image_batch_labels_1 = batch["gen_img_1"]["labels"]

            for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_1, post_gen_image_batch_labels_1):
                if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                    l_gen_img_edit_1 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels)["nll"]
                else:
                    l_gen_img_edit_1 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:])["nll"]
            l_gen_img_edit_1 /= post_gen_image_edit_logits_1.shape[0]
            
            ###########
            post_gen_image_edit_logits_2 = []
            for idx in range(len(batch["gen_img_2"]['labels'])):
                tmp = {}
                for k in batch["gen_img_2"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["gen_img_2"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["gen_img_2"][k][idx]
                post_edit_outputs = edited_model(tmp)
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_gen_image_edit_logits_2.append(post_edit_outputs.logits)
                else:  
                    post_gen_image_edit_logits_2.append(post_edit_outputs)
            post_gen_image_edit_logits_2 = torch.stack(post_gen_image_edit_logits_2, dim=0)
            post_gen_image_batch_labels_2 = batch["gen_img_2"]["labels"]

            for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_2, post_gen_image_batch_labels_2):
                if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                    l_gen_img_edit_2 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels)["nll"]
                else:
                    l_gen_img_edit_2 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:])["nll"]
            l_gen_img_edit_2 /= post_gen_image_edit_logits_2.shape[0]

            ###########
            post_gen_image_edit_logits_3 = []
            for idx in range(len(batch["gen_img_3"]['labels'])):
                tmp = {}
                for k in batch["gen_img_3"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["gen_img_3"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["gen_img_3"][k][idx]
                post_edit_outputs = edited_model(tmp)
                if not isinstance(post_edit_outputs, torch.Tensor):
                    post_gen_image_edit_logits_3.append(post_edit_outputs.logits)
                else:  
                    post_gen_image_edit_logits_3.append(post_edit_outputs)
            post_gen_image_edit_logits_3 = torch.stack(post_gen_image_edit_logits_3, dim=0)
            post_gen_image_batch_labels_3 = batch["gen_img_3"]["labels"]

            for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_3, post_gen_image_batch_labels_3):
                if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                    l_gen_img_edit_3 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels)["nll"]
                else:
                    l_gen_img_edit_3 += self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:])["nll"]
            l_gen_img_edit_3 /= post_gen_image_edit_logits_3.shape[0]

            ################################ REPHRASE IMAGE ################################

            post_image_edit_logits = []
            for idx in range(len(batch["re_image"]['labels'])):
                tmp = {}
                for k in batch["re_image"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["re_image"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["re_image"][k][idx]
                post_image_edit_outputs = edited_model(tmp)
                if not isinstance(post_image_edit_outputs, torch.Tensor):
                    post_image_edit_logits.append(post_image_edit_outputs.logits)
                else:  
                    post_image_edit_logits.append(post_image_edit_outputs)
            post_image_edit_logits = torch.stack(post_image_edit_logits, dim=0)
            post_image_batch_labels = batch["re_image"]["labels"]

            for _post_image_edit_logits, _post_image_batch_labels in zip(post_image_edit_logits, post_image_batch_labels):
                if _post_image_edit_logits.shape[1] > _post_image_batch_labels.shape[1]:    
                    l_image_edit += self.model.edit_loss_fn(self.config, _post_image_edit_logits, _post_image_batch_labels)["nll"]
                else:
                    l_image_edit += self.model.edit_loss_fn(self.config, _post_image_edit_logits, _post_image_batch_labels[:, -_post_image_edit_logits.shape[1]-1:])["nll"]               
            l_image_edit /= post_image_edit_logits.shape[0]

            ################################ RELIABILITY ################################

            inner_edit_outputs = edited_model(batch["edit_inner"])
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs
            inner_batch_labels = batch["edit_inner"]["labels"]

            ####################################################################
            
            # Collect some useful metrics
            with torch.no_grad():

                ################################ RELIABILITY ################################

                # print("origin answer:", batch["edit_inner"]['cond'])
                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])

                ################################ REPHASE IMAGE ################################
  
                rephrase_image_edit_dict = []
                for _post_image_edit_logits, _post_image_batch_labels in zip(post_image_edit_logits, post_image_batch_labels):
                    if _post_image_edit_logits.shape[1] > _post_image_batch_labels.shape[1]:    
                        rephrase_image_edit_dict.append(self.model.edit_loss_fn(self.config, _post_image_edit_logits, _post_image_batch_labels))
                    else:
                        rephrase_image_edit_dict.append(self.model.edit_loss_fn(self.config, _post_image_edit_logits, _post_image_batch_labels[:, -_post_image_edit_logits.shape[1]-1:]))
                
                ################################ GENERALITY ################################
                        
                if post_edit_logits_1.shape[1] > post_batch_labels_1.shape[1]:
                    post_edit_dict_1 = self.model.edit_loss_fn(self.config, post_edit_logits_1, post_batch_labels_1)
                else:
                    post_edit_dict_1 = self.model.edit_loss_fn(self.config, post_edit_logits_1, post_batch_labels_1[:, -post_edit_logits_1.shape[1]-1:])
                
                ###
                post_edit_dict_2 = []
                for _post_edit_logits, _post_batch_labels in zip(post_edit_logits_2, post_batch_labels_2):
                    if _post_edit_logits.shape[1] > _post_batch_labels.shape[1]:
                        post_edit_dict_2.append(self.model.edit_loss_fn(self.config, _post_edit_logits, _post_batch_labels))
                    else:
                        post_edit_dict_2.append(self.model.edit_loss_fn(self.config, _post_edit_logits, _post_batch_labels[:, -_post_edit_logits.shape[1]-1:]))
                
                ###
                if post_edit_logits_3.shape[1] > post_batch_labels_3.shape[1]:
                    post_edit_dict_3 = self.model.edit_loss_fn(self.config, post_edit_logits_3, post_batch_labels_3)
                else:
                    post_edit_dict_3 = self.model.edit_loss_fn(self.config, post_edit_logits_3, post_batch_labels_3[:, -post_edit_logits_3.shape[1]-1:])

                ################################ GENERALITY IMAGE ################################
                        
                post_gen_image_edit_dict_1 = []
                for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_1, post_gen_image_batch_labels_1):
                    if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                        post_gen_image_edit_dict_1.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels))
                    else:
                        post_gen_image_edit_dict_1.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:]))

                post_gen_image_edit_dict_2 = []
                for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_2, post_gen_image_batch_labels_2):
                    if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                        post_gen_image_edit_dict_2.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels))
                    else:
                        post_gen_image_edit_dict_2.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:]))

                post_gen_image_edit_dict_3 = []
                for _post_gen_image_edit_logits, _post_gen_image_batch_labels in zip(post_gen_image_edit_logits_3, post_gen_image_batch_labels_3):
                    if _post_gen_image_edit_logits.shape[1] > _post_gen_image_batch_labels.shape[1]:
                        post_gen_image_edit_dict_3.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels))
                    else:
                        post_gen_image_edit_dict_3.append(self.model.edit_loss_fn(self.config, _post_gen_image_edit_logits, _post_gen_image_batch_labels[:, -_post_gen_image_edit_logits.shape[1]-1:]))
            
            ################################ LOCALITY ################################
                        
            post_base_logits_1=[]; kl_mask_1=[]
            for idx in range(len(batch["loc_1"]['labels'])):
                tmp = {}
                for k in batch["loc_1"]:
                    if k=="image":
                        images = [Image.open(img_path).convert("RGB") for img_path in batch["loc_1"][k][idx]]
                        if self.model_class == "LLaVA":
                            tmp[k] = torch.stack([self.vis_processor(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
                        else:
                            tmp[k] = torch.stack([self.vis_processor(image) for image in images], dim=0)
                    else:
                        tmp[k] = batch["loc_1"][k][idx]
                post_base_outputs = edited_model(tmp)
                if not isinstance(post_base_outputs, torch.Tensor):
                    post_base_logits_1.append(post_base_outputs.logits)
                    kl_mask_1.append(post_base_outputs.attention_mask)
                else:
                    post_base_logits_1.append(post_base_outputs)
                    kl_mask_1.append(torch.ones(post_base_outputs.shape[0], post_base_outputs.shape[1]).to(post_base_outputs.device))
            post_base_logits_1 = torch.stack(post_base_logits_1, dim=0)
            kl_mask_1 = torch.stack(kl_mask_1, dim=0)            

            l_loc_1 = 0.0
            for idx, (_base_logits, _post_base_logits) in enumerate(zip(base_logits_1, post_base_logits_1)):
                l_loc_1 += kl_loc_loss(_base_logits.detach(), _post_base_logits, mask=kl_mask_1[idx])
            l_loc_1 /= post_base_logits_1.shape[0]

            ###
            post_base_outputs = edited_model(batch["loc_2"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits_2 = post_base_outputs.logits
                kl_mask_2 = post_base_outputs.attention_mask
            else:
                post_base_logits_2 = post_base_outputs
                kl_mask_2 = torch.ones(post_base_outputs.shape[0], post_base_outputs.shape[1]).to(post_base_outputs.device)      

            l_loc_2 = kl_loc_loss(base_logits_2.detach(), post_base_logits_2, mask=kl_mask_2[idx])


        if l_edit_1.isnan() or l_edit_2.isnan() or l_edit_3.isnan():
            print("l_edit is nan")
        if l_image_edit.isnan():
            print("l_image_edit is nan")
        if l_loc_1.isnan() or l_loc_2.isnan():
            print("l_loc is nan")

        l_edit = (l_edit_1 + l_edit_2 + l_edit_3)/3
        l_gen_img_edit = (l_gen_img_edit_1 + l_gen_img_edit_2 + l_gen_img_edit_3)/3
        l_loc = (l_loc_1 + l_loc_2)/2

        l_total_edit = self.config.cedit * (l_edit+l_gen_img_edit) + self.config.cloc * l_loc + self.config.iedit * l_image_edit


        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)


        ################################ LOCALITY ################################
            
        post_base_logits_softmax_top_k_2=[]; base_logits_softmax_top_k_2=[]
        for _post_base_logits, _base_logits  in zip(post_base_logits_2, base_logits_2):
            post_base_logits_softmax_top_k_2.append(torch.topk(torch.nn.functional.softmax(_post_base_logits, dim=-1), k=1, dim=-1).indices)
            base_logits_softmax_top_k_2.append(torch.topk(torch.nn.functional.softmax(_base_logits, dim=-1), k=1, dim=-1).indices)
        base_logits_softmax_top_k_2 = torch.stack(base_logits_softmax_top_k_2, dim=0)
        post_base_logits_softmax_top_k_2 = torch.stack(post_base_logits_softmax_top_k_2, dim=0)

        post_base_logits_softmax_top_k_1=[]; base_logits_softmax_top_k_1=[]
        for _post_base_logits, _base_logits  in zip(post_base_logits_1, base_logits_1):
            post_base_logits_softmax_top_k_1.append(torch.topk(torch.nn.functional.softmax(_post_base_logits, dim=-1), k=1, dim=-1).indices)
            base_logits_softmax_top_k_1.append(torch.topk(torch.nn.functional.softmax(_base_logits, dim=-1), k=1, dim=-1).indices)
        base_logits_softmax_top_k_1 = torch.stack(base_logits_softmax_top_k_1, dim=0)
        post_base_logits_softmax_top_k_1 = torch.stack(post_base_logits_softmax_top_k_1, dim=0)

        ################################ INFO DICT ################################
        ### loss ###
        info_dict["time/edit"] = edit_time
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc_1'] = l_loc_1.item()
        info_dict['loss/loc_2'] = l_loc_2.item()

        ### reliability ###
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        rephrase_image_edit_acc = [d["acc"] for d in rephrase_image_edit_dict]
        info_dict['rephrase_image/acc'] = sum(rephrase_image_edit_acc) / len(rephrase_image_edit_acc)

        ### generality ###
        ##
        # info_dict['gen_1_edit/log_prob'] = post_edit_dict_1["log_prob"]
        # info_dict['gen_1_edit/prob'] = post_edit_dict_1["prob"]
        info_dict['gen_1_edit/acc'] = post_edit_dict_1["acc"].item()
        ##
        # post_edit_log_probs_2 = [d["log_prob"] for d in post_edit_dict_2]
        # post_edit_probs_2 = [d["prob"] for d in post_edit_dict_2]
        # info_dict['gen_2_edit/log_prob'] = sum(post_edit_log_probs_2) / len(post_edit_log_probs_2)
        # info_dict['gen_2_edit/prob'] = sum(post_edit_probs_2) / len(post_edit_probs_2)
        cnt1 = 0
        cnt2 = 0
        pred_ids = [d["pred_ids"] for d in post_edit_dict_2]
        trg = [d["targ_ids"] for d in post_edit_dict_2]
        for idx in range(len(pred_ids)):
            if pred_ids[idx][0]==trg[idx][0] and pred_ids[idx][1]==trg[idx][1]:
                cnt1 += 1
            if pred_ids[idx][2]==trg[idx][2] and pred_ids[idx][3]==trg[idx][3]:
                cnt2 += 1
        # info_dict['gen_2_edit/acc'] = cnt / len(pred_ids) / 2
        info_dict['gen_2_edit_1/acc'] = cnt1 / len(pred_ids) 
        info_dict['gen_2_edit_2/acc'] = cnt2 / len(pred_ids) 
        ##
        # info_dict['gen_3_edit/log_prob'] = post_edit_dict_3["log_prob"]
        # info_dict['gen_3_edit/prob'] = post_edit_dict_3["prob"]
        info_dict['gen_3_edit/acc'] = post_edit_dict_3["acc"]

        ### generality_img ###
        ##
        info_dict['gen_img_1_edit/acc'] = sum([d["acc"] for d in post_gen_image_edit_dict_1])/len(post_gen_image_edit_dict_1)
        ##
        cnt = 0
        pred_ids = [d["pred_ids"] for d in post_gen_image_edit_dict_2]
        trg = [d["targ_ids"] for d in post_gen_image_edit_dict_2]
        for idx in range(len(pred_ids)):
            if pred_ids[idx][0]==trg[idx][0] and pred_ids[idx][1]==trg[idx][1]:
                cnt += 1
            if pred_ids[idx][2]==trg[idx][2] and pred_ids[idx][3]==trg[idx][3]:
                cnt += 1
        info_dict['gen_img_2_edit/acc'] = cnt / len(pred_ids) / 2
        ##
        info_dict['gen_img_3_edit/acc'] = sum([d["acc"] for d in post_gen_image_edit_dict_3])/len(post_gen_image_edit_dict_3)

        ### locality ###
        info_dict["loc_1/acc"] = sum(post_base_logits_softmax_top_k_1.view(-1) == base_logits_softmax_top_k_1.view(-1))/post_base_logits_softmax_top_k_1.view(-1).shape[0]
        info_dict["loc_2/acc"] = sum(post_base_logits_softmax_top_k_2.view(-1) == base_logits_softmax_top_k_2.view(-1))/post_base_logits_softmax_top_k_2.view(-1).shape[0]

        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()

        ####################################################################################################
        
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        rephrase_img_acc = f"{stats['rephrase_image/acc_val']:<12.5f}"

        loc_1_acc = f"{stats['loc_1/acc_val']:<12.5f}"
        loc_2_acc = f"{stats['loc_2/acc_val']:<12.5f}"

        gen_1_acc = f"{stats['gen_1_edit/acc_val']:<12.5f}"
        gen_2_acc_1 = f"{stats['gen_2_edit_1/acc_val']:<12.5f}"
        gen_2_acc_2 = f"{stats['gen_2_edit_2/acc_val']:<12.5f}"
        gen_3_acc = f"{stats['gen_3_edit/acc_val']:<12.5f}"

        gen_img_1_acc = f"{stats['gen_img_1_edit/acc_val']:<12.5f}"
        gen_img_2_acc = f"{stats['gen_img_2_edit/acc_val']:<12.5f}"
        gen_img_3_acc = f"{stats['gen_img_3_edit/acc_val']:<12.5f}"

        LOG.info(
          f"Step {prog} inner_acc: {inner_acc} rephrase_image_acc: {rephrase_img_acc} loc_1_acc: {loc_1_acc} loc_2_acc: {loc_2_acc}\
          gen_1_acc: {gen_1_acc} gen_2_acc_1: {gen_2_acc_1} gen_2_acc_2: {gen_2_acc_2} gen_3_acc: {gen_3_acc} \
          gen_1_image_acc: {gen_img_1_acc} gen_2_image_acc: {gen_img_2_acc} gen_3_image_acc: {gen_img_3_acc} \
          it_time: {elapsed:.4f}"
        )


    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats