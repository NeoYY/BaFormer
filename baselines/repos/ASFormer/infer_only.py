"""Inference-only runner for ASFormer using the pretrained epoch-120.model.

Writes MS-TCN recognition files ({vid}.txt) for the shared harness. Mirrors
model.Trainer.predict but skips the PNG/plot side-effects.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from model import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='breakfast')
    ap.add_argument('--split', default='1')
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--model_file', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_rate = 2 if args.dataset == '50salads' else 1
    num_layers, num_f_maps, features_dim = 10, 64, 2048
    channel_mask_rate = 0.3

    ds = os.path.join(args.data_root, args.dataset)
    mapping_file = os.path.join(ds, 'mapping.txt')
    features_path = os.path.join(ds, 'features')
    test_bundle = os.path.join(ds, 'splits', f'test.split{args.split}.bundle')

    with open(mapping_file) as f:
        actions = [ln for ln in f.read().split('\n') if ln.strip()]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    index2label = {v: k for k, v in actions_dict.items()}
    num_classes = len(actions_dict)

    trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
    model = trainer.model
    model.eval().to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))

    with open(test_bundle) as f:
        vids = [ln.strip() for ln in f.readlines() if ln.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        for vid in vids:
            base = vid.split('.')[0]
            feat = np.load(os.path.join(features_path, base + '.npy'))
            feat = feat[:, ::sample_rate]
            x = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)
            preds = model(x, torch.ones(x.size(), device=device))
            _, predicted = torch.max(F.softmax(preds[-1], dim=1).data, 1)
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition.extend([index2label[predicted[i].item()]] * sample_rate)
            with open(os.path.join(args.out_dir, base + '.txt'), 'w') as fp:
                fp.write('### Frame level recognition: ###\n')
                fp.write(' '.join(recognition))
    print(f'wrote {len(vids)} predictions -> {args.out_dir}')


if __name__ == '__main__':
    main()
