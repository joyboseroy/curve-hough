"""Baselines: classical dense Hough on edges, RANSAC fitting, direct
regression, and a DETR-lite hypothesis-query head (matched encoder)."""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from hough import SmallEncoder, curve_pixels


# --------------------------------------------------- classical Hough (edges)
def classical_hough(img, family, size, anchor_bins=32, dense_shapes=24, topn=3):
    """img: [H, W] numpy in [0,1]. Vote binarized edges into (anchor, shape)."""
    edges = (img > 0.5).astype(np.float32)
    ax = np.linspace(0, size - 1, anchor_bins)
    if family == "parabola":
        mag = np.geomspace(0.004, 0.06, dense_shapes // 2)
        shapes = np.concatenate([-mag[::-1], mag])
    else:
        shapes = np.linspace(0.08 * size, 0.35 * size, dense_shapes)
    acc = np.zeros((anchor_bins, anchor_bins, len(shapes)))
    flat = edges.flatten()
    for i, y0 in enumerate(ax):
        for j, x0 in enumerate(ax):
            for k, s in enumerate(shapes):
                idx = curve_pixels(family, (x0, y0, s), size, size, n=200)
                if len(idx):
                    acc[i, j, k] = flat[idx].mean()
    dets = []
    a = acc.copy()
    for _ in range(topn):
        i, j, k = np.unravel_index(a.argmax(), a.shape)
        dets.append((float(ax[j]), float(ax[i]), float(shapes[k]), float(a[i, j, k])))
        a[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = -1  # crude NMS
    return dets


# ---------------------------------------------------------------- RANSAC fit
def ransac(img, family, size, iters=300, tol=2.0, topn=3):
    ys, xs = np.nonzero(img > 0.5)
    pts = np.stack([xs, ys], -1).astype(np.float64)
    dets = []
    rng = np.random.default_rng(0)
    for _ in range(topn):
        if len(pts) < 10:
            break
        best, best_in = None, 0
        for _ in range(iters):
            if family == "parabola":
                s = pts[rng.choice(len(pts), 3, replace=False)]
                A = np.stack([s[:, 0] ** 2, s[:, 0], np.ones(3)], -1)
                try:
                    coef = np.linalg.solve(A, s[:, 1])  # y = c0 x^2 + c1 x + c2
                except np.linalg.LinAlgError:
                    continue
                if abs(coef[0]) < 1e-4:
                    continue
                pred = coef[0] * pts[:, 0] ** 2 + coef[1] * pts[:, 0] + coef[2]
                inl = np.abs(pred - pts[:, 1]) < tol
            else:
                s = pts[rng.choice(len(pts), 3, replace=False)]
                # circle through 3 points
                A = np.stack([2 * s[:, 0], 2 * s[:, 1], np.ones(3)], -1)
                b = (s ** 2).sum(-1)
                try:
                    sol = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                cx, cy = sol[0], sol[1]
                r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
                inl = np.abs(np.linalg.norm(pts - [cx, cy], axis=-1) - r) < tol
                coef = (cx, cy, r)
            if inl.sum() > best_in:
                best, best_in = (coef, inl), inl.sum()
        if best is None:
            break
        coef, inl = best
        if family == "parabola":
            c0, c1, c2 = coef
            x0 = -c1 / (2 * c0)
            y0 = c2 - c0 * x0 ** 2
            dets.append((float(x0), float(y0), float(c0), float(best_in)))
        else:
            dets.append((float(coef[0]), float(coef[1]), float(coef[2]), float(best_in)))
        pts = pts[~inl]
    return dets


# ----------------------------------------------------------- learned baselines
class RegressionHead(nn.Module):
    """Same encoder; global pool; predicts max_curves param vectors + logits."""

    def __init__(self, ch=32, max_curves=3):
        super().__init__()
        self.encoder = SmallEncoder(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, 128), nn.ReLU(),
            nn.Linear(128, max_curves * 4))
        self.max_curves = max_curves

    def forward(self, img):
        z = self.encoder(img).mean((2, 3))
        out = self.mlp(z).view(-1, self.max_curves, 4)
        return out[..., :3], out[..., 3]  # params, presence logits


class QueryHead(nn.Module):
    """DETR-lite: learned hypothesis queries cross-attend encoder tokens."""

    def __init__(self, ch=32, K=8, layers=2):
        super().__init__()
        self.encoder = SmallEncoder(ch)
        self.queries = nn.Parameter(torch.randn(K, ch))
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(ch, 4, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(ch) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, img):
        B = img.shape[0]
        tok = self.encoder(img).flatten(2).transpose(1, 2)     # [B, N, C]
        q = self.queries[None].expand(B, -1, -1)
        for attn, ln in zip(self.blocks, self.norms):
            u, _ = attn(q, tok, tok)
            q = ln(q + u)
        out = self.mlp(q)                                      # [B, K, 4]
        return out[..., :3], out[..., 3]


def hungarian_param_loss(pred, logits, gts, counts):
    """Permutation-matched L1 + BCE presence, per batch element."""
    B = pred.shape[0]
    total = pred.new_zeros(())
    for b in range(B):
        g = gts[b][:counts[b]]
        if counts[b] == 0:
            total = total + nn.functional.binary_cross_entropy_with_logits(
                logits[b], torch.zeros_like(logits[b]))
            continue
        cost = torch.cdist(pred[b], g)                         # [K, k]
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        tgt = torch.zeros_like(logits[b])
        tgt[list(ri)] = 1.0
        total = total + cost[ri, ci].mean() * 0.1 \
            + nn.functional.binary_cross_entropy_with_logits(logits[b], tgt)
    return total / B
