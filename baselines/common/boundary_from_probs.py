"""Score class-agnostic boundary detection from a model's boundary-probability
signal (e.g. ASRF's boundary-regression branch), comparable to BaFormer's head.

For each test video: threshold the boundary prob, keep local maxima as predicted
boundary frames, match against GT segment-change frames within tolerances.
"""
import os
import json
import argparse
import numpy as np

from eval_tas import boundary_positions, boundary_match, prf, load_video_list, read_groundtruth


def peak_positions(prob, threshold):
    p = prob.copy()
    p[p < threshold] = 0.0
    idx = np.where((p[:-2] < p[1:-1]) & (p[2:] < p[1:-1]))[0]
    return (idx + 1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bd_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--model', default='model')
    ap.add_argument('--merge_into', default=None, help='existing result json to add boundary_branch into')
    ap.add_argument('--tols', default='0,1,2,4')
    args = ap.parse_args()

    tols = [int(t) for t in args.tols.split(',')]
    vids = load_video_list(args.bundle)
    micro = {t: [0, 0, 0] for t in tols}
    macro = {t: [] for t in tols}
    for vid in vids:
        prob = np.load(os.path.join(args.bd_dir, vid + '.npy')).astype(np.float64).ravel()
        gt = read_groundtruth(os.path.join(args.gt_dir, vid + '.txt'))
        if len(prob) != len(gt):
            L = min(len(prob), len(gt))
            prob, gt = prob[:L], gt[:L]
        pb = peak_positions(prob, args.threshold)
        gb = boundary_positions(gt)
        for t in tols:
            a, b, c = boundary_match(pb, gb, t)
            micro[t][0] += a; micro[t][1] += b; micro[t][2] += c
            macro[t].append(prf(a, b, c)[2])

    out = {}
    print(f"\n==== {args.model}: boundary-detection from BR branch (th={args.threshold}) ====")
    for t in tols:
        a, b, c = micro[t]
        p, r, f1 = prf(a, b, c)
        mac = float(np.mean(macro[t])) if macro[t] else 0.0
        out[str(t)] = {'precision': p*100, 'recall': r*100,
                       'micro_f1': f1*100, 'macro_f1': mac*100,
                       'tp': a, 'pred': b, 'gt': c}
        print(f"tol={t:>2} | P {p*100:6.2f} | R {r*100:6.2f} | micro-F1 {f1*100:6.2f} | macro-F1 {mac*100:6.2f}")

    if args.merge_into and os.path.isfile(args.merge_into):
        with open(args.merge_into) as f:
            res = json.load(f)
        res['boundary_branch'] = out
        res['boundary'] = out  # use dedicated branch as the reported boundary metric
        res['boundary_source'] = 'BR-branch'
        with open(args.merge_into, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"merged boundary_branch -> {args.merge_into}")


if __name__ == '__main__':
    main()
