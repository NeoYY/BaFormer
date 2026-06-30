"""Full-AttLayer equivalence: original _sliding_window_self_att vs the
vectorized (F.pad + unfold) version. Exercises the real conv_out / window_mask /
scalar_dot_att so we validate the exact tensor the model will produce.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import action_segmentation.models.backbone.asformer_model as A
from action_segmentation.models.backbone.asformer_model import AttLayer

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def opt_sliding(self, q, k, v, mask):
    """Vectorized drop-in for AttLayer._sliding_window_self_att."""
    m, c1, L = q.size()
    _, c2, _ = k.size()
    _, c3, _ = v.size()
    assert m == 1
    bl = self.bl
    nb = L // bl
    rem = L % bl
    if rem != 0:
        pad = bl - rem
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
        nb += 1
    # padding_mask over the (possibly padded) length: 1 for real frames, 0 for pad
    padding_mask = F.pad(mask[:, 0:1, :], (0, nb * bl - L))

    # query windows (already loop-free in the original)
    q = q.reshape(m, c1, nb, bl).permute(0, 2, 1, 3).reshape(m * nb, c1, bl)

    # key/value/mask sliding windows via unfold (replaces python per-window loops)
    w = bl + 2 * (bl // 2)
    p = bl // 2

    def win(t, c):
        return F.pad(t, (p, p)).unfold(2, w, bl).permute(0, 2, 1, 3).reshape(m * nb, c, w)

    k = win(k, c2)
    v = win(v, c3)
    padding_mask = win(padding_mask, 1)
    final_mask = self.window_mask.repeat(m * nb, 1, 1) * padding_mask

    output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
    output = self.conv_out(F.relu(output))
    output = output.reshape(m, nb, -1, bl).permute(0, 2, 1, 3).reshape(m, -1, nb * bl)
    output = output[:, :, 0:L]
    return output * mask[:, 0:1, :]


def main():
    max_d = 0.0
    n = 0
    for bl in [1, 2, 4, 8, 16, 64, 256, 512]:
        layer = AttLayer(64, 64, 64, 2, 2, 2, bl, "encoder", "sliding_att").to(device)
        layer.eval()
        # _sliding_window_self_att receives ALREADY-projected q,k,v (channels = dim//r = 32)
        for L in [1, 3, 7, 100, 1300, 3001, 9000]:
            q = torch.randn(1, 32, L, device=device)
            k = torch.randn(1, 32, L, device=device)
            v = torch.randn(1, 32, L, device=device)
            mask = torch.ones(1, 64, L, device=device)
            mask[:, :, int(L * 0.9):] = 0  # emulate a padded tail
            with torch.no_grad():
                o1 = layer._sliding_window_self_att(q, k, v, mask)
                o2 = opt_sliding(layer, q, k, v, mask)
            assert o1.shape == o2.shape, f"shape bl={bl} L={L}: {o1.shape} vs {o2.shape}"
            d = (o1 - o2).abs().max().item()
            max_d = max(max_d, d)
            n += 1
    print(f"PASS: {n} (bl,L) AttLayer cases. max|orig-opt| = {max_d:.3e}")


if __name__ == "__main__":
    main()
