#%%
from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
import os, argparse, sys

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-enp', '--eval_name_postfix', type=str, default = '', help='Postfix name of this evaluation.')
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [], help='Extra CUDA devices, default empty.')
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='Editor checkpoint path.')
    parser.add_argument('-dn', '--data_name', type=str, default = "AKE", help = 'Evaluating dataset, including EVQA, EIC.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-dp', '--dataset_path', type=str, default = None)
    parser.add_argument('-img_root', '--img_root_dir', type=str, default=4, help = 'Buffer size of data generator.')
    parser.add_argument('-t_loc_image', '--t_loc_image_dir', type=str, default=4, help = 'Buffer size of data generator.')
    parser.add_argument('-dataset_type', '--dataset_type', type=str, default=4, help = 'Buffer size of data generator.')
    
    args = parser.parse_args()
    return args
 

if __name__ == '__main__':
    cfg = get_attr()
    cfg.evaluation_name = cfg.data_name.upper()
    if cfg.eval_name_postfix != '':
        cfg.evaluation_name = '%s-%s'%(cfg.evaluation_name, cfg.eval_name_postfix)
    # if has evaluated, skip
    eval_result_dir_path = os.path.join('eval_results', 'vead', cfg.edit_model_name, cfg.evaluation_name, 'single_edit')
    if os.path.exists(eval_result_dir_path):
        print('Has evaluated: %s'%eval_result_dir_path)
        sys.exit()
    print(cfg)
    
    # load editor
    editor = load_vllm_editor('vead', cfg.edit_model_name, cfg.device, cfg.extra_devices, cfg.editor_ckpt_path, False)
    
    # load data
    if cfg.dataset_type == "AttributeDataset":
        from dataset.ake import AKEVLLMEditData
    elif cfg.dataset_type == "SuppleDataset":
        from dataset.ake_supple import AKEVLLMEditData
    data_path = os.path.join(ROOT_PATH, cfg.dataset_path)
    img_root_dir = os.path.join(ROOT_PATH, cfg.img_root_dir)
    t_loc_image_dir = os.path.join(ROOT_PATH, cfg.t_loc_image_dir)
    eval_data = AKEVLLMEditData(data_path, img_root_dir, t_loc_image_dir, cfg.edit_model_name, cfg.data_sample_n)
   
    # evaluate
    ev = VLLMEditorEvaluation(editor, eval_data, cfg.evaluation_name, 'eval_results')
    results = ev.evaluate_single_edit()

    # If strict AKE, aggregate to AKE-style keys and save alongside
    if isinstance(results, list) and len(results) > 0:
        # Aggregate metrics across samples
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs)/len(xs) if xs else None

        ake_style = {
            'rewrite_acc': None,
            'rephrase_image_acc': None,
            'generality_acc': None,
            'gen1_acc': None,
            'gen2_acc': None,
            'generality_image_acc': None,
            'locality_acc': None,
            'sample_count': len(results)
        }

        # rewrite_acc
        ake_style['rewrite_acc'] = mean([r['reliability'].get('acc') for r in results])

        # rephrase_image_acc (use rephrase_image bucket)
        img_ref_accs = []
        for r in results:
            for it in r['generality'].get('rephrase_image', []):
                img_ref_accs.append(it.get('acc'))
        ake_style['rephrase_image_acc'] = mean(img_ref_accs)

        # generality_acc (text) = combine Gen_1/2/3 s* (exclude text_rephrase to match AKE)
        text_gen_accs = []
        re_img_accs, gen1_accs, gen2_accs = [], [], []
        buckets = ['rephrase_image','gen1','gen2']
        for r in results:
            for b in buckets:
                for it in r['generality'].get(b, []):
                    text_gen_accs.append(it.get('acc'))
            for it in r['generality'].get('rephrase_image', []): re_img_accs.append(it.get('acc'))
            for it in r['generality'].get('gen1', []): gen1_accs.append(it.get('acc'))
            for it in r['generality'].get('gen2', []): gen2_accs.append(it.get('acc'))
        ake_style['generality_acc'] = mean(text_gen_accs)
        ake_style['gen1_acc'] = mean(gen1_accs)
        ake_style['gen2_acc'] = mean(gen2_accs)

        # generality_image_acc = rephrase_image (already computed)
        ake_style['generality_image_acc'] = ake_style['rephrase_image_acc']

        # locality_acc = mean of both text_loc and image_loc
        loc_accs = []
        for r in results:
            for it in r['locality'].get('loc_m', []):
                loc_accs.append(it.get('acc'))
            for it in r['locality'].get('loc_t', []):
                loc_accs.append(it.get('acc'))
        ake_style['locality_acc'] = mean(loc_accs)

        # save next to mean_results.json
        out_dir = os.path.join('eval_results', 'vead', editor.name_of_editor_and_model()[1], 'single_edit')
        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(os.path.join(out_dir, 'ake_mean_results.json'), 'w') as f:
            json.dump(ake_style, f, indent=4)

 