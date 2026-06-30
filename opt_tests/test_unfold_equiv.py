"""Numerical-equivalence test: original list-comprehension sliding-window
construction vs a vectorized torch.unfold version.

We replicate ONLY the windowing of k, v, padding_mask (the part that differs);
scalar_dot_att is unchanged, so if the windowed tensors are bit-identical the
whole attention is identical.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


def original_windows(k, v, padding_mask, bl, nb):
    # mirrors AttLayer._sliding_window_self_att exactly
    k = torch.cat([torch.zeros(1, k.size(1), bl // 2, device=k.device), k,
                   torch.zeros(1, k.size(1), bl // 2, device=k.device)], dim=-1)
    v = torch.cat([torch.zeros(1, v.size(1), bl // 2, device=v.device), v,
                   torch.zeros(1, v.size(1), bl // 2, device=v.device)], dim=-1)
    padding_mask = torch.cat([torch.zeros(1, 1, bl // 2, device=k.device), padding_mask,
                              torch.zeros(1, 1, bl // 2, device=k.device)], dim=-1)
    k = torch.cat([k[:, :, i * bl:(i + 1) * bl + (bl // 2) * 2] for i in range(nb)], dim=0)
    v = torch.cat([v[:, :, i * bl:(i + 1) * bl + (bl // 2) * 2] for i in range(nb)], dim=0)
    padding_mask = torch.cat(
        [padding_mask[:, :, i * bl:(i + 1) * bl + (bl // 2) * 2] for i in range(nb)], dim=0)
    return k, v, padding_mask


def unfold_windows(k, v, padding_mask, bl, nb):
    w = bl + 2 * (bl // 2)
    p = bl // 2

    def win(t):
        t = F.pad(t, (p, p))                      # pad last dim by bl//2 each side
        t = t.unfold(2, w, bl)                    # (1, c, nb, w)
        return t.permute(0, 2, 1, 3).reshape(1 * nb, t.size(1), w)
    return win(k), win(v), win(padding_mask)


def make_inputs(c, L, bl):
    # emulate the state right before windowing: L already padded to multiple of bl
    nb = L // bl
    if L % bl != 0:
        nb += 1
    Lp = nb * bl
    k = torch.randn(1, c, Lp, device=device)
    v = torch.randn(1, c, Lp, device=device)
    padding_mask = torch.ones(1, 1, Lp, device=device)
    # make some of the mask zero (padding region) like the real code
    real_L = L
    padding_mask[:, :, real_L:] = 0.0
    return k, v, padding_mask, nb


def main():
    c = 32
    Ls = [1, 2, 3, 5, 7, 100, 1300, 9001]
    bls = [1, 2, 4, 8, 16, 64, 256, 512]
    max_abs = 0.0
    ncases = 0
    for L in Ls:
        for bl in bls:
            k, v, pm, nb = make_inputs(c, L, bl)
            ok = original_windows(k, v, pm, bl, nb)
            uf = unfold_windows(k, v, pm, bl, nb)
            for a, b, name in zip(ok, uf, ["k", "v", "mask"]):
                assert a.shape == b.shape, f"shape mismatch L={L} bl={bl} {name}: {a.shape} vs {b.shape}"
                d = (a - b).abs().max().item()
                max_abs = max(max_abs, d)
                assert d == 0.0, f"value mismatch L={L} bl={bl} {name}: max|d|={d}"
            ncases += 1
    print(f"PASS: {ncases} (L,bl) cases, all windows bit-identical. max|diff|={max_abs}")


if __name__ == "__main__":
    main()
