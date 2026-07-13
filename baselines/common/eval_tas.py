"""Shared temporal-action-segmentation evaluator.

Replicates BaFormer's metric definitions (action_segmentation/utils/metrics.py and
Inference.py) so that external baselines are scored on identical footing:

  * Acc          : frame-level micro accuracy (correct frames / total frames) * 100
  * Edit         : mean over videos of normalised Levenshtein (segment) edit score
  * F1@{10,25,50}: global tp/fp/fn accumulated over all videos, single F1 each
  * Class-agnostic boundary detection P / R / micro-F1 / macro-F1 at tolerances
    {0,1,2,4} frames. Boundaries are the frame indices where the (predicted or GT)
    label sequence changes -- i.e. the start of every new action segment, exactly
    matching BaFormer's `frame_bd` ground truth.

Predictions are read from MS-TCN "recognition" files: a first header line followed
by a line of space-separated class *names*, one per frame (the format written by
MS-TCN / ASFormer / DiffAct / ASRF / C2F-TCN). A plain single-line file is also
accepted.
"""
import os
import json
import argparse
import numpy as np


# --------------------------- segment helpers (BaFormer) ---------------------------
def get_labels_start_end_time(frame_wise_labels, bg_class):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)  # BaFormer convention
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros((m_row + 1, n_col + 1), dtype=np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for j in range(n_col + 1):
        D[0, j] = j
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1 - D[-1, -1] / max(m_row, n_col)) * 100
    return D[-1, -1]


def edit_score(recognized, ground_truth, bg_class):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, True)


def f_score(recognized, ground_truth, overlap, bg_class):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


# --------------------------- class-agnostic boundaries ---------------------------
def boundary_positions(labels):
    """Frame indices where the label sequence changes (start of a new segment)."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def boundary_match(pred_idx, gt_idx, tol):
    """Greedy one-to-one matching within +/- tol frames (BaFormer Inference.py)."""
    gt_used = [False] * len(gt_idx)
    tp = 0
    for p in pred_idx:
        best_j, best_d = -1, tol + 1
        for j, g in enumerate(gt_idx):
            if gt_used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
    return tp, len(pred_idx), len(gt_idx)


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_gt if n_gt > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# --------------------------- IO ---------------------------
def read_recognition(path):
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]
    lines = [ln for ln in lines if ln.strip() != '']
    if len(lines) >= 2 and lines[0].lstrip().startswith('###'):
        return lines[1].split()
    if len(lines) == 1:
        return lines[0].split()
    # header + possibly multi-line: take everything after first header line
    if lines and lines[0].lstrip().startswith('###'):
        return ' '.join(lines[1:]).split()
    return ' '.join(lines).split()


def read_groundtruth(path):
    with open(path, 'r') as f:
        return [ln.strip() for ln in f.readlines() if ln.strip() != '']


def evaluate(pred_dir, gt_dir, video_list, bg_class, tols=(0, 1, 2, 4)):
    overlap = [.1, .25, .5]
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)
    correct = 0
    total = 0
    edit_sum = 0.0
    n_vid = 0

    bd_micro = {t: [0, 0, 0] for t in tols}   # tp, n_pred, n_gt
    bd_macro = {t: [] for t in tols}

    for vid in video_list:
        gt = read_groundtruth(os.path.join(gt_dir, f'{vid}.txt'))
        pred = read_recognition(os.path.join(pred_dir, f'{vid}.txt'))
        if len(pred) != len(gt):
            L = min(len(pred), len(gt))
            pred, gt = pred[:L], gt[:L]
        n_vid += 1
        for i in range(len(gt)):
            total += 1
            if gt[i] == pred[i]:
                correct += 1
        edit_sum += edit_score(pred, gt, bg_class)
        for s in range(3):
            a, b, c = f_score(pred, gt, overlap[s], bg_class)
            tp[s] += a
            fp[s] += b
            fn[s] += c
        # boundary detection
        pb = boundary_positions(pred)
        gb = boundary_positions(gt)
        for t in tols:
            a, b, c = boundary_match(pb, gb, t)
            bd_micro[t][0] += a
            bd_micro[t][1] += b
            bd_micro[t][2] += c
            bd_macro[t].append(prf(a, b, c)[2])

    acc = 100.0 * correct / total
    edit = edit_sum / n_vid
    f1s = []
    for s in range(3):
        p = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        r = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(np.nan_to_num(f1) * 100)

    boundary = {}
    for t in tols:
        a, b, c = bd_micro[t]
        p, r, f1 = prf(a, b, c)
        macro = float(np.mean(bd_macro[t])) if bd_macro[t] else 0.0
        boundary[str(t)] = {
            'precision': p * 100, 'recall': r * 100,
            'micro_f1': f1 * 100, 'macro_f1': macro * 100,
            'tp': a, 'pred': b, 'gt': c,
        }

    return {
        'n_videos': n_vid,
        'Acc': acc,
        'Edit': edit,
        'F1@10': f1s[0], 'F1@25': f1s[1], 'F1@50': f1s[2],
        'boundary': boundary,
    }


def load_video_list(bundle_path):
    with open(bundle_path, 'r') as f:
        vids = [ln.strip() for ln in f.readlines() if ln.strip() != '']
    return [v[:-4] if v.endswith('.txt') else v for v in vids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--bundle', required=True, help='test.splitN.bundle path')
    ap.add_argument('--dataset', default='breakfast')
    ap.add_argument('--model', default='model')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    bg_class = [] if args.dataset in ('breakfast', '50salads') else ['SIL']
    video_list = load_video_list(args.bundle)
    res = evaluate(args.pred_dir, args.gt_dir, video_list, bg_class)
    res['model'] = args.model
    res['dataset'] = args.dataset

    print(f"\n==== {args.model} on {args.dataset} ({res['n_videos']} videos) ====")
    print(f"Acc   : {res['Acc']:.2f}")
    print(f"Edit  : {res['Edit']:.2f}")
    print(f"F1@10 : {res['F1@10']:.2f}")
    print(f"F1@25 : {res['F1@25']:.2f}")
    print(f"F1@50 : {res['F1@50']:.2f}")
    print("---- class-agnostic boundary detection ----")
    for t, b in res['boundary'].items():
        print(f"tol={t:>2} | P {b['precision']:6.2f} | R {b['recall']:6.2f} | "
              f"micro-F1 {b['micro_f1']:6.2f} | macro-F1 {b['macro_f1']:6.2f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"\nsaved -> {args.out_json}")


if __name__ == '__main__':
    main()
