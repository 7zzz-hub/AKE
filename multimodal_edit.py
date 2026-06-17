import argparse
from easyeditor import  MultimodalTrainer
from easyeditor import AttributeDataset, SuppleDataset
from easyeditor import FTMultimodalHparams, MENDMultimodalTrainingHparams, MENDMultimodalHparams, SERACMultimodalTrainingHparams, SERACMultimodalHparams

def test_FT(args):
    hparams = FTMultimodalHparams.from_hparams(args.config_path)

    for key, value in vars(args).items():
        if key != 'config_path':
            setattr(hparams, key, value)
            
    if args.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(args.eval_json_path, size=args.size, config=hparams)
    elif args.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(args.eval_json_path, size=args.size, config=hparams)
        
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def train_MEND(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams(args.config_path)

    for key, value in vars(args).items():
        if key != 'config_path':
            setattr(hparams, key, value)
            
    if args.dataset_type == 'SuppleDataset':
        train_ds = SuppleDataset(args.train_json_path, size=args.size, config=hparams)
        eval_ds = SuppleDataset(args.eval_json_path, size=args.size, config=hparams)
    elif args.dataset_type == 'AttributeDataset':
        train_ds = AttributeDataset(args.train_json_path, size=args.size, config=hparams, eval_mode=False)
        eval_ds = AttributeDataset(args.eval_json_path, size=args.size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def test_MEND(args):
    hparams = MENDMultimodalHparams.from_hparams(args.config_path)

    for key, value in vars(args).items():
        if key != 'config_path':
            setattr(hparams, key, value)
            
    if args.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(args.eval_json_path, size=args.size, config=hparams)
    elif args.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(args.eval_json_path, size=args.size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run() 

def train_SERAC(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams(args.config_path)

    for key, value in vars(args).items():
        if key != 'config_path':
            setattr(hparams, key, value)
            
    if args.dataset_type == 'SuppleDataset':
        train_ds = SuppleDataset(args.train_json_path, size=args.size, config=hparams)
        eval_ds = SuppleDataset(args.eval_json_path, size=args.size, config=hparams)
    elif args.dataset_type == 'AttributeDataset':
        train_ds = AttributeDataset(args.train_json_path, size=args.size, config=hparams, eval_mode=False)
        eval_ds = AttributeDataset(args.eval_json_path, size=args.size, config=hparams)

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def test_SERAC(args):
    hparams = SERACMultimodalHparams.from_hparams(args.config_path)

    for key, value in vars(args).items():
        if key != 'config_path':
            setattr(hparams, key, value)
            
    if args.dataset_type == 'SuppleDataset':
        eval_ds = SuppleDataset(args.eval_json_path, size=args.size, config=hparams)
    elif args.dataset_type == 'AttributeDataset':
        eval_ds = AttributeDataset(args.eval_json_path, size=args.size, config=hparams)

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
    parser.add_argument("--dataset_type", type=str, required=True, choices=["AttributeDataset", "SuppleDataset"])
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--m_loc_image", type=str, required=False)
    parser.add_argument("--t_loc_image", type=str, required=False)
    
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    
    
    args = parser.parse_args()

    # FT
    if args.method=="FT-L" or args.method=="FT-V":
        test_FT(args)
    
    # MEND
    elif args.method=="MEND" and args.mode=="train":
        train_MEND(args)
    elif args.method=="MEND" and args.mode=="eval":
        test_MEND(args)

    # SERAC
    elif args.method=="SERAC" and args.mode=="train":
        train_SERAC(args)
    elif args.method=="SERAC" and args.mode=="eval":
        test_SERAC(args)