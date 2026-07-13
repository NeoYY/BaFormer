"""Aggregate baseline result JSONs into a comparison table (Breakfast split1)."""
import os
import json
import glob
import argparse


ORDER = ['DiffAct', 'ASFormer', 'LTContext', 'C2F-TCN', 'ASRF', 'BaFormer']


def sort_key(model):
    for i, k in enumerate(ORDER):
        if k.lower() in model.lower():
            return i
    return len(ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='baselines/results')
    ap.add_argument('--out_md', default='baselines/results/COMPARISON.md')
    args = ap.parse_args()

    rows = []
    for jf in glob.glob(os.path.join(args.results_dir, '*.json')):
        with open(jf) as f:
            r = json.load(f)
        if 'Acc' not in r:
            continue
        rows.append(r)
    rows.sort(key=lambda r: sort_key(r.get('model', '')))

    seg_hdr = f"| {'Model':<22} | {'Acc':>6} | {'Edit':>6} | {'F1@10':>6} | {'F1@25':>6} | {'F1@50':>6} |"
    seg_sep = "|" + "-" * 24 + "|" + ("-" * 8 + "|") * 5
    print("\n### Segmentation metrics (standard MS-TCN eval, Breakfast split1)\n")
    print(seg_hdr)
    print(seg_sep)
    seg_lines = [seg_hdr, seg_sep]
    for r in rows:
        line = (f"| {r['model']:<22} | {r['Acc']:>6.2f} | {r['Edit']:>6.2f} | "
                f"{r['F1@10']:>6.2f} | {r['F1@25']:>6.2f} | {r['F1@50']:>6.2f} |")
        print(line)
        seg_lines.append(line)

    print("\n### Class-agnostic boundary detection (micro-F1 %, from segmentation transitions unless noted)\n")
    bd_hdr = f"| {'Model':<22} | {'tol=0':>6} | {'tol=1':>6} | {'tol=2':>6} | {'tol=4':>6} |"
    bd_sep = "|" + "-" * 24 + "|" + ("-" * 8 + "|") * 4
    print(bd_hdr)
    print(bd_sep)
    bd_lines = [bd_hdr, bd_sep]
    for r in rows:
        b = r.get('boundary', {})
        def f(t):
            return b.get(str(t), {}).get('micro_f1', float('nan'))
        note = r.get('boundary_source', '')
        name = r['model'] + (f" ({note})" if note else '')
        line = (f"| {name:<22} | {f(0):>6.2f} | {f(1):>6.2f} | {f(2):>6.2f} | {f(4):>6.2f} |")
        print(line)
        bd_lines.append(line)

    with open(args.out_md, 'w') as f:
        f.write("# Breakfast split1 — baseline comparison\n\n")
        f.write("## Segmentation metrics (standard MS-TCN eval)\n\n")
        f.write("\n".join(seg_lines) + "\n\n")
        f.write("## Class-agnostic boundary detection (micro-F1 %)\n\n")
        f.write("\n".join(bd_lines) + "\n")
    print(f"\nsaved -> {args.out_md}")


if __name__ == '__main__':
    main()
