import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

import argparse

from easyeditor import  MultimodalTrainer
from easyeditor import AttributeDataset, SuppleDataset
from easyeditor import FTMultimodalHparams, MENDMultimodalTrainingHparams, MENDMultimodalHparams, SERACMultimodalTrainingHparams, SERACMultimodalHparams

def test_FT(hparams_path, eval_json_path, size=None):
    hparams = FTMultimodalHparams.from_hparams(hparams_path)

    if hparams.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif hparams.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)
        
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def train_MEND(hparams_path, train_json_path, eval_json_path, size=None):
    hparams = MENDMultimodalTrainingHparams.from_hparams(hparams_path)

    if hparams.dataset_type == 'SuppleDataset':
        train_ds = SuppleDataset(train_json_path, size=size, config=hparams)
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif hparams.dataset_type == 'AttributeDataset':
        train_ds = AttributeDataset(train_json_path, size=size, config=hparams, eval_mode=False)
        eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def test_MEND(hparams_path, eval_json_path, size=None):
    hparams = MENDMultimodalHparams.from_hparams(hparams_path)

    if hparams.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif hparams.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run() 

def train_SERAC(hparams_path, train_json_path, eval_json_path, size=None):
    hparams = SERACMultimodalTrainingHparams.from_hparams(hparams_path)

    if hparams.dataset_type == 'SuppleDataset':
        train_ds = SuppleDataset(train_json_path, size=size, config=hparams)
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif hparams.dataset_type == 'AttributeDataset':
        train_ds = AttributeDataset(train_json_path, size=size, config=hparams, eval_mode=False)
        eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def test_SERAC(hparams_path, eval_json_path, size=None):
    hparams = SERACMultimodalHparams.from_hparams(hparams_path)

    if hparams.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif hparams.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(eval_json_path, size=size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run() 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["llava", "blip2", "qwen-vl"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["MEND", "SERAC", "FT-L", "FT-V"])
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--train_json_path", required=True)
    parser.add_argument("--eval_json_path", required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"])
    args = parser.parse_args()

    # FT
    if args.method=="FT-L" or args.method=="FT-V":
        test_FT(args.config_path, args.eval_json_path, args.size)
    
    # MEND
    elif args.method=="MEND" and args.mode=="train":
        train_MEND(args.config_path, args.train_json_path, args.eval_json_path, args.size)
    elif args.method=="MEND" and args.mode=="eval":
        test_MEND(args.config_path, args.eval_json_path, args.size)

    # SERAC
    elif args.method=="SERAC" and args.mode=="train":
        train_SERAC(args.config_path, args.train_json_path, args.eval_json_path, args.size)
    elif args.method=="SERAC" and args.mode=="eval":
        test_SERAC(args.config_path, args.eval_json_path, args.size)