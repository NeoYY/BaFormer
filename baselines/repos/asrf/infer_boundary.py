"""ASRF inference: dump boundary-refined recognition files + boundary probs.

Loads a trained ASRF checkpoint and, for each Breakfast test video, writes:
  * {out_dir}/{video}.txt          -- MS-TCN recognition (boundary-refined preds)
  * {bd_dir}/{video}.npy           -- per-frame boundary probability (BR branch)
so the shared harness can score segmentation metrics and we can separately score
the class-agnostic boundary-detection metric from ASRF's dedicated boundary branch.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.class_id_map import get_n_classes, get_id2class_map
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.postprocess import PostProcessor
from libs.transformer import TempDownSamp, ToTensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--bd_dir', required=True)
    args = ap.parse_args()

    config = get_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    downsamp_rate = 2 if config.dataset == '50salads' else 1

    data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode='test', split=config.split,
        dataset_dir=config.dataset_dir, csv_dir=config.csv_dir)
    loader = DataLoader(data, batch_size=1, shuffle=False,
                        num_workers=config.num_workers, collate_fn=collate_fn)

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)
    id2class = get_id2class_map(config.dataset, dataset_dir=config.dataset_dir)

    model = models.ActionSegmentRefinementFramework(
        in_channel=config.in_channel, n_features=config.n_features,
        n_classes=n_classes, n_stages=config.n_stages, n_layers=config.n_layers,
        n_stages_asb=config.n_stages_asb, n_stages_brb=config.n_stages_brb)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    postprocessor = PostProcessor('refinement_with_boundary', config.boundary_th)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.bd_dir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for sample in loader:
            x = sample['feature'].to(device)
            path = sample['feature_path'][0]
            name = os.path.basename(path)[:-4]
            mask = sample['mask'].numpy()

            output_cls, output_bound = model(x)
            output_cls = output_cls.to('cpu').data.numpy()
            output_bound = output_bound.to('cpu').data.numpy()

            refined = postprocessor(output_cls, boundaries=output_bound, masks=mask)[0]
            names = [id2class[int(i)] for i in refined]
            with open(os.path.join(args.out_dir, name + '.txt'), 'w') as f:
                f.write('### Frame level recognition: ###\n')
                f.write(' '.join(names))
            np.save(os.path.join(args.bd_dir, name + '.npy'), output_bound[0, 0])
            n += 1
    print(f'wrote {n} predictions -> {args.out_dir} ; boundary probs -> {args.bd_dir}')


if __name__ == '__main__':
    main()
