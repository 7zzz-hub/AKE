#%%
from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor
import os, argparse


def get_attr():
    def parse_lkpt(value:str):
        if value.lower() == 'none':
            return None
        return value
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help = 'Train dataset sample number.', required = True)
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    # other settings
    parser.add_argument('-dsn', '--data_n', type=int, default=None, help = 'Train dataset sample number.')
    parser.add_argument('-lkpt', '--load_ckpt_path', type=parse_lkpt, default = None, help='Editor checkpoint path.')
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [0], help='Extra CUDA devices, default empty.')
    parser.add_argument('-eps', '--epochs', type=int, default=100, help = 'Train epochs.')
    parser.add_argument('-tnp', '--train_name_prefix', type=str, default=None, help = 'Train name prefix.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=100, help = 'Save checkpoint per iteraions.')
    parser.add_argument('-lpi', '--log_per_i', type=int, default=1, help = 'Log per iteraions.')
    parser.add_argument('-ea', '--ema_alpha', type=float, default=0.1, help = 'EMA loss alpha.')
    parser.add_argument('-rs', '--random_seed', type=int, default=None, help = 'Random seed.')
    parser.add_argument('-dbs', '--data_buffer_size', type=int, default=4, help = 'Buffer size of data generator.')
    parser.add_argument('-dp', '--dataset_path', type=str, default=4, help = 'Buffer size of data generator.')
    parser.add_argument('-img_root', '--img_root_dir', type=str, default=4, help = 'Buffer size of data generator.')
    parser.add_argument('-t_loc_image', '--t_loc_image', type=str, default=None, help = 'Buffer size of data generator.')
    parser.add_argument('-dataset_type', '--dataset_type', type=str, default=4, help = 'Buffer size of data generator.')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = get_attr()

    # load editor
    editor = load_vllm_editor('vead', cfg.edit_model_name, cfg.device, cfg.extra_devices, None, True)
    
    # load data
    if cfg.dataset_type == "AttributeDataset":
        from dataset.ake import AKEVLLMEditData
    elif cfg.dataset_type == "SuppleDataset":
        from dataset.ake_supple import AKEVLLMEditData
    data_path = os.path.join(ROOT_PATH, cfg.dataset_path)
    img_root_dir = os.path.join(ROOT_PATH, cfg.img_root_dir)
    t_loc_image  = os.path.join(ROOT_PATH, cfg.t_loc_image )
    
    train_data = AKEVLLMEditData(data_path, img_root_dir, t_loc_image, cfg.edit_model_name, cfg.data_n)

    # initialize and train
    editor.train_init(train_data, cfg.batch_size, train_name_prefix = cfg.train_name_prefix,
        load_ckpt_path = cfg.load_ckpt_path, save_ckpt_per_i = cfg.save_ckpt_per_i, 
        log_per_i = cfg.log_per_i, ema_alpha = cfg.ema_alpha, random_seed = cfg.random_seed,
        data_buffer_size = cfg.data_buffer_size)
    editor.train(cfg.epochs)

 


