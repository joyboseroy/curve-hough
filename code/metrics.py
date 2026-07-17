"""Curve-EA score and Hungarian-matched precision/recall/F evaluation."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def sample_curve(family, p, size, n=200):
    if family == "parabola":
        x0, y0, a = p
        xs = np.linspace(0, size - 1, n)
        ys = a * (xs - x0) ** 2 + y0
        tang = np.stack([np.ones_like(xs), 2 * a * (xs - x0)], -1)
    else:
        x0, y0, r = p
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = x0 + r * np.cos(t), y0 + r * np.sin(t)
        tang = np.stack([-np.sin(t), np.cos(t)], -1)
    m = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    pts = np.stack([xs, ys], -1)[m]
    tang = tang[m]
    tang = tang / (np.linalg.norm(tang, axis=-1, keepdims=True) + 1e-9)
    return pts, tang


def curve_ea(family, p1, p2, size):
    pts1, t1 = sample_curve(family, p1, size)
    pts2, t2 = sample_curve(family, p2, size)
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0
    d12 = np.linalg.norm(pts1[:, None] - pts2[None], axis=-1)
    nn12, nn21 = d12.argmin(1), d12.argmin(0)
    chamfer = 0.5 * (d12.min(1).mean() + d12.min(0).mean())
    dmax = 0.25 * size * np.sqrt(2)
    Sd = max(0.0, 1.0 - chamfer / dmax)
    cosang = np.abs((t1 * t2[nn12]).sum(-1)).clip(0, 1)
    ang = np.arccos(cosang)  # in [0, pi/2] due to abs
    Sth = max(0.0, 1.0 - ang.mean() / (np.pi / 2))
    return float((Sd * Sth) ** 2)


def prf(dets_per_img, gts_per_img, family, size, thresholds=None, return_sims=False):
    """dets: list per image of (x0, y0, s, score); gts: list per image of params."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    P, R, F = [], [], []
    sims_all, n_det, n_gt = [], 0, 0
    for dets, gts in zip(dets_per_img, gts_per_img):
        n_det += len(dets)
        n_gt += len(gts)
        if not dets or not gts:
            continue
        M = np.zeros((len(dets), len(gts)))
        for i, d in enumerate(dets):
            for j, g in enumerate(gts):
                M[i, j] = curve_ea(family, d[:3], g, size)
        ri, ci = linear_sum_assignment(-M)
        sims_all.extend(M[ri, ci].tolist())
    sims_all = np.array(sims_all)
    for tau in thresholds:
        tp = (sims_all >= tau).sum()
        p = tp / max(n_det, 1)
        r = tp / max(n_gt, 1)
        P.append(p)
        R.append(r)
        F.append(2 * p * r / max(p + r, 1e-9))
    result = (float(np.mean(P)), float(np.mean(R)), float(np.mean(F)))
    if return_sims:
        return result, sims_all
    return result
