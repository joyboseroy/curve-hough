"""Synthetic parametric-curve benchmark: parabolas, circles, ellipses, lanes,
and lines, with noise, occlusion, and distractor segments. Images are HxW
float32 in [0,1]."""
import numpy as np
import torch
from torch.utils.data import Dataset


def rasterize_points(img, xs, ys, width=2):
    H, W = img.shape
    for dx in range(-(width // 2), width // 2 + 1):
        for dy in range(-(width // 2), width // 2 + 1):
            xi = np.clip(np.round(xs + dx).astype(int), 0, W - 1)
            yi = np.clip(np.round(ys + dy).astype(int), 0, H - 1)
            img[yi, xi] = 1.0
    return img


def draw_parabola(img, x0, y0, a, n=600):
    H, W = img.shape
    xs = np.linspace(0, W - 1, n)
    ys = a * (xs - x0) ** 2 + y0
    m = (ys >= 0) & (ys < H)
    return rasterize_points(img, xs[m], ys[m])


def draw_lane(img, y0, x0, a, n=600):
    """Near-vertical parabola: x = a(y-y0)^2 + x0."""
    H, W = img.shape
    ys = np.linspace(0, H - 1, n)
    xs = a * (ys - y0) ** 2 + x0
    m = (xs >= 0) & (xs < W)
    return rasterize_points(img, xs[m], ys[m])


def draw_circle(img, x0, y0, r, n=600):
    t = np.linspace(0, 2 * np.pi, n)
    xs, ys = x0 + r * np.cos(t), y0 + r * np.sin(t)
    H, W = img.shape
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return rasterize_points(img, xs[m], ys[m])


def draw_ellipse(img, x0, y0, rx, ry, phi, n=600):
    t = np.linspace(0, 2 * np.pi, n)
    ct, st = np.cos(phi), np.sin(phi)
    xs = x0 + rx * np.cos(t) * ct - ry * np.sin(t) * st
    ys = y0 + rx * np.cos(t) * st + ry * np.sin(t) * ct
    H, W = img.shape
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return rasterize_points(img, xs[m], ys[m])


def draw_line(img, theta, r, n=600):
    """x*cos(theta) + y*sin(theta) = r, origin at image center.
    Matches hough.py's curve_pixels 'line' branch exactly, so a model
    trained on this generator sees the same geometry it'll be scored on."""
    H, W = img.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ct, st = np.cos(theta), np.sin(theta)
    if abs(st) > abs(ct):
        xs = np.linspace(0, W - 1, n)
        ys = (r - (xs - cx) * ct) / st + cy
    else:
        ys = np.linspace(0, H - 1, n)
        xs = (r - (ys - cy) * st) / ct + cx
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return rasterize_points(img, xs[m], ys[m])


def draw_segment(img, p, q, n=200):
    xs = np.linspace(p[0], q[0], n)
    ys = np.linspace(p[1], q[1], n)
    return rasterize_points(img, xs, ys, width=2)


FAMILY_PARAM_LEN = {"parabola": 3, "circle": 3, "ellipse": 5, "lane": 3, "line": 2}


class SyntheticCurves(Dataset):
    """family: 'parabola' -> (x0, y0, a); 'circle' -> (x0, y0, r);
    'ellipse' -> (x0, y0, rx, ry, phi); 'lane' -> (y0, x0, a);
    'line' -> (theta, r). Param tensor is always width 5, padded with
    zeros beyond each family's actual length."""

    def __init__(self, family="parabola", size=128, n_images=8000, seed=0,
                 noise=0.08, occlude=True, distractors=True, max_curves=3):
        self.family, self.size, self.n = family, size, n_images
        self.noise, self.occlude, self.distractors = noise, occlude, distractors
        self.max_curves = max_curves
        self.seed = seed
        self.param_len = FAMILY_PARAM_LEN[family]

    def __len__(self):
        return self.n

    def sample_params(self, rng):
        s = self.size
        out = np.zeros(5)
        if self.family == "parabola":
            out[0] = rng.uniform(0.15 * s, 0.85 * s)
            out[1] = rng.uniform(0.15 * s, 0.85 * s)
            out[2] = rng.choice([-1, 1]) * rng.uniform(0.004, 0.06)
        elif self.family == "lane":
            out[0] = rng.uniform(0.15 * s, 0.85 * s)   # y0
            out[1] = rng.uniform(0.15 * s, 0.85 * s)   # x0
            out[2] = rng.choice([-1, 1]) * rng.uniform(0.004, 0.06)
        elif self.family == "circle":
            out[0] = rng.uniform(0.2 * s, 0.8 * s)
            out[1] = rng.uniform(0.2 * s, 0.8 * s)
            out[2] = rng.uniform(0.08 * s, 0.35 * s)
        elif self.family == "ellipse":
            out[0] = rng.uniform(0.25 * s, 0.75 * s)
            out[1] = rng.uniform(0.25 * s, 0.75 * s)
            out[2] = rng.uniform(0.10 * s, 0.30 * s)
            out[3] = rng.uniform(0.06 * s, 0.22 * s)
            out[4] = rng.uniform(0, np.pi)
        else:  # line
            diag = np.sqrt(s ** 2 + s ** 2) / 2
            out[0] = rng.uniform(0, np.pi)             # theta
            out[1] = rng.uniform(-0.8 * diag, 0.8 * diag)  # r, kept off the
            # extreme edge of the range so the line reliably crosses the frame
        return out

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1000003 + idx)
        s = self.size
        img = np.zeros((s, s), np.float32)
        k = rng.integers(1, self.max_curves + 1)
        params = np.zeros((self.max_curves, 5), np.float32)
        for j in range(k):
            p = self.sample_params(rng)
            params[j] = p
            if self.family == "parabola":
                draw_parabola(img, p[0], p[1], p[2])
            elif self.family == "lane":
                draw_lane(img, p[0], p[1], p[2])
            elif self.family == "circle":
                draw_circle(img, p[0], p[1], p[2])
            elif self.family == "ellipse":
                draw_ellipse(img, p[0], p[1], p[2], p[3], p[4])
            else:
                draw_line(img, p[0], p[1])
        if self.distractors:
            for _ in range(rng.integers(0, 4)):
                draw_segment(img, rng.uniform(0, s, 2), rng.uniform(0, s, 2))
        if self.occlude:
            for _ in range(rng.integers(0, 3)):
                x, y = rng.integers(0, s, 2)
                w, h = rng.integers(8, s // 3, 2)
                img[y:y + h, x:x + w] = 0.0
        img = np.clip(img + rng.normal(0, self.noise, img.shape), 0, 1).astype(np.float32)
        return torch.from_numpy(img)[None], torch.from_numpy(params), int(k)
