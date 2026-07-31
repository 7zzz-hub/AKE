from typing import Optional
import argparse, os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import get_full_model_name, load_vllm_editor
from utils.GLOBAL import ROOT_PATH
from dataset.ake import AKEVLLMEditData
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    # model & device
    parser.add_argument('-mn', '--model_name', type=str, required=True, help='Model name part: llava / blip2 / qwen3vl')
    parser.add_argument('-dvc', '--device', type=str, required=True, help='CUDA device, e.g. "cuda:0"')
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default=[], help='Extra CUDA devices')
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default=None, help='Editor checkpoint path')

    # data
    parser.add_argument('-split', '--split', type=str, required=True, choices=['train', 'val1', 'val2'],
                        help='AKE split to evaluate')
    parser.add_argument('-dr', '--data_root', type=str, required=True,
                        help='Path to AKE data root (contains data/llava, data/blip2)')
    parser.add_argument('-ir', '--img_root', type=str, required=True,
                        help='Root directory for CLEVR images')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default=None, help='Sample number for evaluation')
    parser.add_argument('--strict_ake', action='store_true', help='Enable strict alignment with original AKE metrics')

    # naming
    parser.add_argument('-enp', '--eval_name_postfix', type=str, default='', help='Postfix for evaluation name')
    return parser.parse_args()


def get_ake_json_path(data_root: str, model_name_part: str, split: str) -> str:
    model_name_part = model_name_part.lower()
    if 'llava' in model_name_part:
        sub = 'llava'
    elif 'blip2' in model_name_part or 'blip' in model_name_part:
        sub = 'blip2'
    elif 'qwen' in model_name_part:
        sub = 'qwenvl'
    else:
        raise ValueError('Unsupported model_name for AKE path: %s' % model_name_part)
    fname = {
        'train': 'train_dataset.json',
        'val1': 'val1_dataset.json',
        'val2': 'val2_dataset.json',
    }[split]
    return os.path.join(data_root, 'data', sub, fname)


if __name__ == '__main__':
    cfg = parse_args()
    full_model_name = get_full_model_name(cfg.model_name)
    evaluation_name = 'AKE-%s' % cfg.split
    if cfg.eval_name_postfix:
        evaluation_name = '%s-%s' % (evaluation_name, cfg.eval_name_postfix)

    # load editor (VEAD) with underlying VLLM
    editor = load_vllm_editor('vead', full_model_name, cfg.device, cfg.extra_devices,
                              cfg.editor_ckpt_path, False)

    # build dataset from AKE JSON
    ake_json = get_ake_json_path(cfg.data_root, cfg.model_name, cfg.split)
    # BLIP-2 needs leading space in targets; LLaVA not
    add_leading_space = ('blip' in cfg.model_name.lower())
    eval_data = AKEVLLMEditData(ake_json, cfg.img_root, cfg.data_sample_n,
                                add_leading_space_for_targets=add_leading_space)

    # evaluate
    ev = VLLMEditorEvaluation(editor, eval_data, evaluation_name, 'eval_results')
    results = ev.evaluate_single_edit()

    # If strict AKE, aggregate to AKE-style keys and save alongside
    if cfg.strict_ake and isinstance(results, list) and len(results) > 0:
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
            'gen3_acc': None,
            'generality_image_acc': None,
            'locality_acc': None,
            'sample_count': len(results)
        }

        # rewrite_acc
        ake_style['rewrite_acc'] = mean([r['reliability'].get('acc') for r in results])

        # rephrase_image_acc (use image_rephrase bucket)
        img_ref_accs = []
        for r in results:
            for it in r['generality'].get('image_rephrase', []):
                img_ref_accs.append(it.get('acc'))
        ake_style['rephrase_image_acc'] = mean(img_ref_accs)

        # generality_acc (text) = combine Gen_1/2/3 s* (exclude text_rephrase to match AKE)
        text_gen_accs = []
        gen1_accs, gen2_accs, gen3_accs = [], [], []
        buckets = ['Gen_1_s1','Gen_1_s2','Gen_1_s3','Gen_2_s1','Gen_2_s2','Gen_2_s3','Gen_3_s1','Gen_3_s2','Gen_3_s3']
        for r in results:
            for b in buckets:
                for it in r['generality'].get(b, []):
                    text_gen_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_1_s1', []): gen1_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_1_s2', []): gen1_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_1_s3', []): gen1_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_2_s1', []): gen2_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_2_s2', []): gen2_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_2_s3', []): gen2_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_3_s1', []): gen3_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_3_s2', []): gen3_accs.append(it.get('acc'))
            for it in r['generality'].get('Gen_3_s3', []): gen3_accs.append(it.get('acc'))
        ake_style['generality_acc'] = mean(text_gen_accs)
        ake_style['gen1_acc'] = mean(gen1_accs)
        ake_style['gen2_acc'] = mean(gen2_accs)
        ake_style['gen3_acc'] = mean(gen3_accs)

        # generality_image_acc = image_rephrase (already computed)
        ake_style['generality_image_acc'] = ake_style['rephrase_image_acc']

        # locality_acc = mean of both text_loc and image_loc
        loc_accs = []
        for r in results:
            for it in r['locality'].get('text_loc', []):
                loc_accs.append(it.get('acc'))
            for it in r['locality'].get('image_loc', []):
                loc_accs.append(it.get('acc'))
        ake_style['locality_acc'] = mean(loc_accs)

        # save next to mean_results.json
        out_dir = os.path.join('eval_results', 'vead', editor.name_of_editor_and_model()[1], evaluation_name, 'single_edit')
        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(os.path.join(out_dir, 'ake_mean_results.json'), 'w') as f:
            json.dump(ake_style, f, indent=4)


