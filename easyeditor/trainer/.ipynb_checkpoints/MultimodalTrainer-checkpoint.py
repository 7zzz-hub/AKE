from .BaseTrainer import *
import logging
import time
import copy

import torch
from .losses import kl_loc_loss
from torch.utils.data import Dataset
from .utils import (
    RunningStatAverager,
    safe_backward,
)
from tqdm import tqdm
from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers import AutoProcessor
from .qa import answer_single_question, compute_single_score, prepare_inputs, edit_loc_data

Log = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        # get tokenizer and vis_processor
        if config.model_class in ["Blip2OPT", "minigpt4"]:
            self.vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            self.vis_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        elif config.model_class == "qwen-vl":
            self.vis_processor = AutoProcessor.from_pretrained(config.name)
        elif "owl-2" in config.model_name.lower():
            self.vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")
        
        self.config = config
        self.model_class = config.model_class
        self.device = config.device
        
        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

    def edit_step(self, batch, training: bool):

        record = {
            'rel':{}, 'Loc_in':{},'Loc_out':{},
            're_image':{},'gen1':{},'gen2':{},
        }

        ##############
        # PRE-EDIT
        ##############

        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            
            # Loc_in
            base_logits_1, _ = answer_single_question(self.config, self.vis_processor, self.model, batch["Loc_in"])
            compute_single_score(self.config, self.model, base_logits_1, _, batch["Loc_in"], record['Loc_in'])

            # Loc_out
            base_logits_2, _ = answer_single_question(self.config, self.vis_processor, self.model, batch["Loc_out"])
            compute_single_score(self.config, self.model, base_logits_2, _, batch["Loc_out"], record['Loc_out'])

        ###############
        # POST-EDIT
        ###############
                
        # Do the edit
        start = time.time()

        edit_inner = prepare_inputs(self.config, self.vis_processor, batch["rel"][0])
        edited_model, model_info = self.model.edit(edit_inner)
        
        edit_time = time.time() - start
              
        info_dict = {}
        l_total, l_image_edit, l_loc, l_base = 0, 0, 0, 0, 
        
        with torch.set_grad_enabled(training):

            ############## RELIABILITY ##############

            inner_edit_logits, inner_batch_labels = answer_single_question(self.config, self.vis_processor, edited_model, batch["rel"])

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:    
                l_edit = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)["nll"]
            else:
                l_edit = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])["nll"] 
            
            ############## REPHRASED IMAGE ##############
            
            post_image_edit_logits, post_image_batch_labels = answer_single_question(self.config, self.vis_processor, edited_model, batch["re_image"])
         
            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:     
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
            else:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])["nll"]               

            ########
            
            # Collect some useful metrics
            with torch.no_grad():
            
                # rel
                inner_edit_dict = compute_single_score(self.config, self.model, inner_edit_logits, inner_batch_labels, 
                                                       batch['rel'], record['rel'])

                # rephrase
                rephrase_image_edit_dict = compute_single_score(self.config, self.model, post_image_edit_logits, post_image_batch_labels, 
                                                               batch['re_image'], record['re_image'])
                
                # gen
                if not training:
                    logits, labels = answer_single_question(self.config, self.vis_processor, edited_model, batch["gen1"])
                    gen1_dict = compute_single_score(self.config, self.model, logits, labels, batch["gen1"], record['gen1'])
                    logits, labels = answer_single_question(self.config, self.vis_processor, edited_model, batch["gen2"])
                    gen2_dict = compute_single_score(self.config, self.model, logits, labels, batch["gen2"], record['gen2'])

            ################################ LOCALITY ################################
                
            l_Loc_in, post_base_logits_1 = edit_loc_data(self.config, self.vis_processor, edited_model, kl_loc_loss, base_logits_1, batch["Loc_in"])
            l_Loc_out, post_base_logits_2 = edit_loc_data(self.config, self.vis_processor, edited_model, kl_loc_loss, base_logits_2, batch["Loc_out"])


        ###############
        # LOSS
        ###############
            
        if l_edit.isnan():
            print("l_edit is nan")
        if l_image_edit.isnan():
            print("l_image_edit is nan")
        if l_Loc_in.isnan() or l_Loc_out.isnan():
            print("l_loc is nan")
        
        # l_loc = (l_Loc_in + l_Loc_out)/2
        # l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit

        # if self.config.alg == "SERAC_MULTI":
        #     l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_Loc_out + self.config.iedit * l_image_edit
        # else:
        
        l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_Loc_in+l_Loc_out) + self.config.iedit * l_image_edit
        
        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        ################################ LOCALITY ################################

        post_base_topk_2 = torch.stack([
            torch.topk(torch.nn.functional.softmax(_post_base_logits, dim=-1), k=1, dim=-1).indices
            for _post_base_logits in post_base_logits_2
        ], dim=0)

        base_topk_2 = torch.stack([
            torch.topk(torch.nn.functional.softmax(_base_logits, dim=-1), k=1, dim=-1).indices
            for _base_logits in base_logits_2
        ], dim=0)

        post_base_topk_1 = torch.stack([
            torch.topk(torch.nn.functional.softmax(_post_base_logits, dim=-1), k=1, dim=-1).indices
            for _post_base_logits in post_base_logits_1
        ], dim=0)

        base_topk_1 = torch.stack([
            torch.topk(torch.nn.functional.softmax(_base_logits, dim=-1), k=1, dim=-1).indices
            for _base_logits in base_logits_1
        ], dim=0)


        ################################ INFO DICT ################################

        ### loss ###
        info_dict["time/edit"] = edit_time
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/Loc_in'] = l_Loc_in.item()
        info_dict['loss/Loc_out'] = l_Loc_out.item()


        ### reliability ###
        record['rel']['acc'] = info_dict['inner/acc'] = inner_edit_dict["acc"]
        record['rel']['exact_match_acc'] = inner_edit_dict['exact_match_acc']
        record['re_image']['acc'] = info_dict['rephrase_image/acc'] = rephrase_image_edit_dict["acc"]
        record['re_image']['exact_match_acc'] = rephrase_image_edit_dict['exact_match_acc']
        
        ## generality ###
        def compute_acc_gen2(pred_list, targ_list):
            if pred_list[0]==targ_list[0] and pred_list[1]==targ_list[1]:
                return 1.0
            return 0.0 
        
        if not training:
            record['gen1']['acc'] = info_dict['gen1/acc'] = gen1_dict['acc']
            record['gen1']['exact_match_acc'] = gen1_dict['exact_match_acc']
            if self.config.dataset_type == "AttributeDataset":
                record['gen2']['acc'] = info_dict['gen2/acc'] = compute_acc_gen2(gen2_dict['targ_token'], gen2_dict['pred_token'])
            else:
                record['gen2']['acc'] = info_dict['gen2/acc'] = gen2_dict['exact_match_acc']
        
        
        ### locality ###
        record['Loc_in']['acc'] = info_dict["Loc_in/acc"] = sum(post_base_topk_1.view(-1) == base_topk_1.view(-1))/post_base_topk_1.view(-1).shape[0]
        record['Loc_in']['acc'] = info_dict["Loc_out/acc"] = sum(post_base_topk_2.view(-1) == base_topk_2.view(-1))/post_base_topk_2.view(-1).shape[0]

        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_inax"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()

        ########################################
        
        info_dict = {**info_dict}
        
        return l_total, l_edit, l_loc, l_base, info_dict, record

    def train_step(self, batch):

        l_total, l_edit, l_loc, l_base, info_dict, _ = self.edit_step(
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

        Loc_in_acc = f"{stats['Loc_in/acc_val']:<12.5f}"
        Loc_out_acc = f"{stats['Loc_out/acc_val']:<12.5f}"
        
        gen1 = f"{stats.get('gen1/acc_val'):<12.5f}"
        gen2 = f"{stats.get('gen2/acc_val'):<12.5f}"


        Log.info(
          f"Step {prog} inner_acc: {inner_acc} Loc_in_acc: {Loc_in_acc} Loc_out_acc: {Loc_out_acc}\
          rephrase_image_acc: {rephrase_img_acc} gen1: {gen1} gen2: {gen2}  \
          it_time: {elapsed:.4f}"
        )


    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            Log.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        records = []
        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            _, _, _, _, info_dict, record = self.edit_step(batch, training=False)
            records.append(record)

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

        return stats, records