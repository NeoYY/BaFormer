"""Micro-benchmark for the BaFormer model forward+backward (batch_size=1).

The per-step cost is data-independent (kernel-launch bound), so we drive the
real model with synthetic features of realistic sequence lengths. This isolates
exactly what the optimizations target (model fwd+bwd) and is reused to measure
before/after each change. End-to-end timing on real data is measured separately
during actual training.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action_segmentation import create_model, get_default_config, update_config


def build_config(dataset_dir="/tmp/none/"):
    cfg = get_default_config()
    cfg.merge_from_file("configs/framed_en_de.yaml", allow_unsafe=True)
    cfg.merge_from_list([
        "dataset.name", "breakfast",
        "dataset.split", 1,
        "dataset.dataset_dir", dataset_dir,
        "device", "cuda",
    ])
    cfg = update_config(cfg)
    cfg.freeze()
    return cfg


def sum_tensor_leaves(obj):
    """A scalar surrogate loss that backprops through every float output."""
    total = 0.0
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.float().mean()
        return 0.0
    if isinstance(obj, dict):
        for v in obj.values():
            total = total + sum_tensor_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            total = total + sum_tensor_leaves(v)
    return total


def synth_batch(L, device):
    # I3D features are (1, 2048, L); values ~N(0,1) like normalized features
    return torch.randn(1, 2048, L, device=device)


def time_steps(model, optimizer, L, n_warmup, n_iter, device, train=True):
    fwd_ms, bwd_ms, step_ms = [], [], []
    for it in range(n_warmup + n_iter):
        data = synth_batch(L, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = sum_tensor_leaves(outputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if train:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if it >= n_warmup:
            fwd_ms.append((t1 - t0) * 1e3)
            bwd_ms.append((t2 - t1) * 1e3)
            step_ms.append((t2 - t0) * 1e3)
    return np.array(fwd_ms), np.array(bwd_ms), np.array(step_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1300])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--eval", action="store_true", help="forward-only (no backward)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    cfg = build_config()
    model = create_model(cfg).to(device)
    if args.eval:
        model.eval()
    else:
        model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"device={torch.cuda.get_device_name(0)} params={nparams/1e6:.2f}M "
          f"compile={args.compile} mode={'eval' if args.eval else 'train'}")

    if args.compile:
        model = torch.compile(model)

    print(f"{'L':>7} | {'fwd ms':>18} | {'bwd ms':>18} | {'step ms':>18} | {'vid/s':>7}")
    for L in args.lengths:
        fwd, bwd, step = time_steps(model, optimizer, L, args.warmup, args.iters,
                                    device, train=not args.eval)
        print(f"{L:>7} | {fwd.mean():7.2f} +-{fwd.std():5.2f} | "
              f"{bwd.mean():7.2f} +-{bwd.std():5.2f} | "
              f"{step.mean():7.2f} +-{step.std():5.2f} | {1000.0/step.mean():7.2f}")


if __name__ == "__main__":
    main()
