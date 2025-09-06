import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

import argparse
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import AttributeDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys


####################### MiniGPT4 ##########################
def train_MEND_MiniGPT4(train_json_path, eval_json_path):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def test_MEND_MiniGPT4(eval_json_path):
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def train_SERAC_MiniGPT4(train_json_path, eval_json_path):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def test_SERAC_MiniGPT4(eval_json_path):
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def test_FT_MiniGPT4(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_MiniGPT4_Qformer(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_MiniGPT4(eval_json_path, size=None):
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True        
    )
    
    print_result(metrics, save_path='results/IKE/MiniGPT4_results_portability.txt')


####################### BLIP2 ##########################
def train_MEND_Blip2OPT(train_json_path, eval_json_path, size=None):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = AttributeDataset(train_json_path, size=size, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def test_MEND_Blip2OPT(eval_json_path, size=None):
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()    

    
def train_SERAC_Blip2OPT(train_json_path, eval_json_path, size=None):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = AttributeDataset(train_json_path, size=size, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_SERAC_Blip2OPT(eval_json_path, size=None):
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Blip2OPT(eval_json_path, size=None):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Blip2OPT_QFormer(eval_json_path, size=None):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


####################### LLAVA ##########################
def train_MEND_LLaVA(train_json_path, eval_json_path, size=None):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/llava.yaml')
    train_ds = AttributeDataset(train_json_path, size=size,config=hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size,config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()    

def test_MEND_LLaVA(eval_json_path, size=None):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def train_SERAC_LLaVA(train_json_path, eval_json_path, size=None):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/llava.yaml')
    train_ds = AttributeDataset(train_json_path, size=size,config=hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size,config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()    

def test_SERAC_LLaVA(eval_json_path, size=None):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size,config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   

def test_FT_LLaVA(eval_json_path,size=None):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size,config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_LLaVA_mmproj(eval_json_path,size=None):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = AttributeDataset(eval_json_path, size=size,config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


####################### Qwen-VL ##########################
def train_MEND_QwenVL(train_json_path, eval_json_path):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/qwenvl.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_MEND_QwenVL(eval_json_path):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/qwenvl.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def train_SERAC_QwenVL(train_json_path, eval_json_path):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/qwenvl.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()    

def test_SERAC_QwenVL(eval_json_path):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/qwenvl.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   

def test_FT_QwenVL(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwenvl.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_QwenVL_ViT(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwenvl_vit.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


####################### Owl-2 ##########################
def train_MEND_Owl2(train_json_path, eval_json_path):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/owl2.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_MEND_Owl2(eval_json_path):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/owl2.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def train_SERAC_Owl2(train_json_path, eval_json_path):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/owl2.yaml')
    train_ds = AttributeDataset(train_json_path, config=hparams)
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()    

def test_SERAC_Owl2(eval_json_path):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/owl2.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   

def test_FT_Owl2(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Owl2_Visual(eval_json_path):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2_visual.yaml')
    eval_ds = AttributeDataset(eval_json_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_IKE_Owl2(eval_json_path, size=None):
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/owl2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True        
    )
    
    print_result(metrics, save_path=os.path.join(hparams.results_dir,'IKE/Owl2_results_portability.txt'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["llava", "qwen", "blip2", "minigpt4", "owl"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["IKE", "MEND", "SERAC", "FT-L", "FT-V", "ROME"])
    parser.add_argument("--train_json_path", type=str, default='data/blip2/train_dataset.json')
    parser.add_argument("--eval1_json_path", type=str, default='data/blip2/val1_dataset.json')
    parser.add_argument("--eval2_json_path", type=str, default='data/blip2/val2_dataset_tmp.json')
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    #SERAC
    if args.model == "llava" and args.method == "SERAC":
        train_SERAC_LLaVA(train_json_path=args.train_json_path, eval_json_path=args.eval1_json_path, size=args.size)
        test_SERAC_LLaVA(eval_json_path=args.eval2_json_path, size=args.size)
    elif args.model == "blip2" and args.method == "SERAC":
        train_SERAC_Blip2OPT(train_json_path=args.train_json_path, eval_json_path=args.eval1_json_path, size=args.size)
        test_SERAC_Blip2OPT(eval_json_path=args.eval2_json_path, size=args.size)
    
    #MEND
    elif args.model == "llava" and args.method == "MEND":
        train_MEND_LLaVA(train_json_path=args.train_json_path, eval_json_path=args.eval1_json_path, size=args.size)
        test_MEND_LLaVA(eval_json_path=args.eval1_json_path, size=args.size)
    elif args.model == "blip2" and args.method == "MEND":
        train_MEND_Blip2OPT(train_json_path=args.train_json_path, eval_json_path=args.eval1_json_path, size=args.size)
        # test_MEND_Blip2OPT(eval_json_path=args.eval2_json_path, size=args.size)

    #FT
    elif args.model == "llava" and args.method == "FT-L":
        test_FT_LLaVA(eval_json_path=args.eval1_json_path, size=args.size)
    elif args.model == "llava" and args.method == "FT-V":
        test_FT_LLaVA_mmproj(eval_json_path=args.eval1_json_path, size=args.size)

    elif args.model == "blip2" and args.method == "FT-L":
        test_FT_Blip2OPT(eval_json_path=args.eval1_json_path, size=args.size)
    elif args.model == "blip2" and args.method == "FT-V":
        test_FT_Blip2OPT_QFormer(eval_json_path=args.eval1_json_path, size=args.size)
        
    else:
        raise ValueError(f"未知模型: {args.model}")
