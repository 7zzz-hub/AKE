import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

import argparse

from easyeditor import  MultimodalTrainer
from easyeditor import AttributeDataset, SuppleDataset
# from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, MENDMultimodalHparams \
#     , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import FTMultimodalHparams

def test_FT(hparams_path, eval_json_path, dataset_type, size=None):
    hparams = FTMultimodalHparams.from_hparams(hparams_path)

    if dataset_type == 'Supple':
        eval_ds = SuppleDataset(eval_json_path, size=size, config=hparams)
    elif dataset_type == 'AttributeDataset':
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
    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--train_json_path", required=True)
    parser.add_argument("--eval_json_path", required=True)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"])
    args = parser.parse_args()

    #SERAC
    if args.method=="FT-L" or args.method=="FT-V":
        test_FT(args.config_path, args.eval_json_path, args.dataset_type, args.size)