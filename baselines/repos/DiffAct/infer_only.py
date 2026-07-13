"""Inference-only runner for DiffAct using the shipped pretrained release.model.

Loads a Breakfast split config, builds the test dataset, restores the pretrained
weights, and writes MS-TCN style recognition files (one per test video) that the
shared harness (baselines/common/eval_tas.py) can score. Deliberately avoids
DiffAct's own func_eval (uses removed np.float under numpy>=1.24).
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from dataset import get_data_dict, VideoFeatureDataset
from utils import load_config_file
from main import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--mode', default='decoder-agg')
    args = ap.parse_args()

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    p = load_config_file(args.config)

    root = p['root_data_dir']
    name = p['dataset_name']
    split_id = p['split_id']
    feature_dir = os.path.join(root, name, 'features')
    label_dir = os.path.join(root, name, 'groundTruth')
    mapping_file = os.path.join(root, name, 'mapping.txt')

    event_list = [i[1] for i in np.loadtxt(mapping_file, dtype=str)]

    test_video_list = np.loadtxt(os.path.join(
        root, name, 'splits', f'test.split{split_id}.bundle'), dtype=str)
    test_video_list = [i.split('.')[0] for i in test_video_list]

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, label_dir=label_dir,
        video_list=test_video_list, event_list=event_list,
        sample_rate=p['sample_rate'], temporal_aug=p['temporal_aug'],
        boundary_smooth=p['boundary_smooth'])
    test_dataset = VideoFeatureDataset(test_data_dict, len(event_list), mode='test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(dict(p['encoder_params']), dict(p['decoder_params']),
                      dict(p['diffusion_params']), event_list, p['sample_rate'],
                      p['temporal_aug'], p['set_sampling_seed'], p['postprocess'],
                      device=device)
    trainer.model.load_state_dict(torch.load(args.model_path, map_location=device))
    trainer.model.eval().to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            video, pred, label = trainer.test_single_video(
                idx, test_dataset, args.mode, device, model_path=None)
            pred_names = [event_list[int(i)] for i in pred]
            with open(os.path.join(args.out_dir, f'{video}.txt'), 'w') as f:
                f.write('### Frame level recognition: ###\n')
                f.write(' '.join(pred_names))
    print(f'wrote {len(test_dataset)} predictions -> {args.out_dir}')


if __name__ == '__main__':
    main()
