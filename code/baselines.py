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
    if family == "ellipse":
        return ransac_ellipse(img, size, tol=tol, topn=topn)
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


# ------------------------------------------------- ellipse RANSAC (5-pt conic)
def _conic_fit(P):
    """Algebraic conic fit A x^2 + B xy + C y^2 + D x + E y + F = 0 through
    >=5 points (null vector of the design matrix). Returns coefficients."""
    x, y = P[:, 0], P[:, 1]
    Dm = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], -1)
    return np.linalg.svd(Dm, full_matrices=True)[2][-1]


def _conic_to_ellipse(c):
    """Conic coefficients -> (cx, cy, a1, a2, phi) with a1 the semi-axis along
    direction phi, matching dataset.draw_ellipse's (rx, ry, phi); None if the
    conic is not a real ellipse."""
    A, B, C, D, E, F = c
    if B * B - 4 * A * C >= 0:
        return None
    M = np.array([[A, B / 2], [B / 2, C]])
    try:
        ctr = -0.5 * np.linalg.solve(M, np.array([D, E]))
    except np.linalg.LinAlgError:
        return None
    Fp = F + 0.5 * (D * ctr[0] + E * ctr[1])
    lam, V = np.linalg.eigh(M)
    if np.any(np.abs(lam) < 1e-12):          # numerically degenerate conic
        return None
    q = -Fp / lam
    if np.any(q <= 0):
        return None
    a = np.sqrt(q)
    phi = float(np.arctan2(V[1, 0], V[0, 0]) % np.pi)
    return float(ctr[0]), float(ctr[1]), float(a[0]), float(a[1]), phi


def _sampson(c, P):
    x, y = P[:, 0], P[:, 1]
    A, B, C, D, E, F = c
    f = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    fx, fy = 2 * A * x + B * y + D, B * x + 2 * C * y + E
    return np.abs(f) / (np.sqrt(fx * fx + fy * fy) + 1e-12)


def ransac_ellipse(img, size, iters=3000, tol=2.0, topn=3, axis_range=(0.04, 0.45)):
    """RANSAC with the minimal 5-point conic solver, Sampson inlier distance,
    least-squares refit on the inliers, and sequential extraction of up to
    topn ellipses. Candidate ellipses with a semi-axis outside axis_range*size
    are rejected (the benchmark's ellipses are well inside this range).
    iters is larger than for the 3-point families because a clean 5-point
    sample from one of several curves is much rarer."""
    ys, xs = np.nonzero(img > 0.5)
    pts = np.stack([xs, ys], -1).astype(np.float64)
    lo, hi = axis_range[0] * size, axis_range[1] * size
    rng = np.random.default_rng(0)
    dets = []
    for _ in range(topn):
        if len(pts) < 20:
            break
        mu, sc = pts.mean(0), pts.std() + 1e-9
        Q = (pts - mu) / sc
        best_in, best_inl = 0, None
        for _ in range(iters):
            c = _conic_fit(Q[rng.choice(len(Q), 5, replace=False)])
            e = _conic_to_ellipse(c)
            if e is None:
                continue
            if not (lo <= e[2] * sc <= hi and lo <= e[3] * sc <= hi):
                continue
            inl = _sampson(c, Q) * sc < tol
            if inl.sum() > best_in:
                best_in, best_inl = int(inl.sum()), inl
        if best_inl is None or best_in < 15:
            break
        c = _conic_fit(Q[best_inl])                     # refit on all inliers
        e = _conic_to_ellipse(c)
        if e is None or not (lo <= e[2] * sc <= hi and lo <= e[3] * sc <= hi):
            break
        cx, cy, a1, a2, phi = e
        dets.append((float(mu[0] + sc * cx), float(mu[1] + sc * cy),
                     float(a1 * sc), float(a2 * sc), phi, float(best_in)))
        pts = pts[~(_sampson(c, Q) * sc < tol)]
    return dets


# ----------------------------------------------------------- learned baselines
class RegressionHead(nn.Module):
    """Same encoder; global pool; predicts max_curves param vectors + logits."""

    def __init__(self, ch=32, max_curves=3, n_params=3):
        super().__init__()
        self.encoder = SmallEncoder(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, 128), nn.ReLU(),
            nn.Linear(128, max_curves * (n_params + 1)))
        self.max_curves, self.n_params = max_curves, n_params

    def forward(self, img):
        z = self.encoder(img).mean((2, 3))
        out = self.mlp(z).view(-1, self.max_curves, self.n_params + 1)
        return out[..., :self.n_params], out[..., self.n_params]  # params, presence logits


class QueryHead(nn.Module):
    """DETR-lite: learned hypothesis queries cross-attend encoder tokens."""

    def __init__(self, ch=32, K=8, layers=2, n_params=3):
        super().__init__()
        self.encoder = SmallEncoder(ch)
        self.queries = nn.Parameter(torch.randn(K, ch))
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(ch, 4, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(ch) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.Linear(ch, 64), nn.ReLU(), nn.Linear(64, n_params + 1))
        self.n_params = n_params

    def forward(self, img):
        B = img.shape[0]
        tok = self.encoder(img).flatten(2).transpose(1, 2)     # [B, N, C]
        q = self.queries[None].expand(B, -1, -1)
        for attn, ln in zip(self.blocks, self.norms):
            u, _ = attn(q, tok, tok)
            q = ln(q + u)
        out = self.mlp(q)                                      # [B, K, n_params+1]
        return out[..., :self.n_params], out[..., self.n_params]


FAMILY_SCALE = {
    # per-dim scale so cdist/L1 cost weighs position and shape comparably.
    # (x0, y0 are pixel-scale; parabola 'a' is tiny, circle 'r' is pixel-scale)
    "parabola": torch.tensor([1.0, 1.0, 0.03]),
    "circle": torch.tensor([1.0, 1.0, 1.0]),
    # ellipse (x0, y0, rx, ry, phi): phi in radians is scaled so ~1 rad weighs
    # about 10 px. Note the parametrization is not unique ((rx, ry, phi) and
    # (ry, rx, phi + pi/2) draw the same ellipse), which hurts direct regression;
    # that is a property of the baseline, not something tuned away here.
    "ellipse": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1]),
}
FAMILY_NPARAMS = {"parabola": 3, "circle": 3, "ellipse": 5}


def hungarian_param_loss(pred, logits, gts, counts, family="parabola", pos_weight=4.0):
    """Permutation-matched L1 + BCE presence, per batch element.
    Parameters are normalized per-dimension before matching/loss so that a
    tiny shape parameter (e.g. parabola curvature) isn't drowned out by
    pixel-scale position error, and presence BCE uses pos_weight to offset
    the class imbalance between K queries and a few true curves."""
    B = pred.shape[0]
    scale = FAMILY_SCALE[family].to(pred.device)
    total = pred.new_zeros(())
    pw = torch.tensor(pos_weight, device=pred.device)
    for b in range(B):
        # gts are zero-padded to width 5 (dataset.py, since ellipse support);
        # keep only this family's real parameter columns (3 for parabola/circle)
        g = gts[b][:counts[b], :scale.shape[0]]
        if counts[b] == 0:
            total = total + nn.functional.binary_cross_entropy_with_logits(
                logits[b], torch.zeros_like(logits[b]), pos_weight=pw)
            continue
        cost = torch.cdist(pred[b] / scale, g / scale)          # [K, k]
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        tgt = torch.zeros_like(logits[b])
        tgt[list(ri)] = 1.0
        total = total + cost[ri, ci].mean() * 0.1 \
            + nn.functional.binary_cross_entropy_with_logits(
                logits[b], tgt, pos_weight=pw)
    return total / B
