"""RANSAC line-fit baseline on real TuSimple images, for the 'line' family.

Runs on the EXACT same held-out test split train_real.py used (same seed=0
shuffle over the same RealTuSimpleDataset construction), so the comparison
against the trained model's F/similarity numbers is apples-to-apples --
not a different subset that happens to look favorable either way.

No deep model, no training: classical edge detection (Sobel) + RANSAC line
fitting on the edge pixels, scored with the same curve-EA metric and
Hungarian matching as everything else in this project.

Usage:
    python real_data/ransac_line_baseline.py \
        --root /content/tusimple/TUSimple/train_set --max-images 4000
"""
import argparse
import sys
import os
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from metrics import prf
from tusimple_dataset import RealTuSimpleDataset


def edge_pixels(img, thresh_percentile=90):
    """Sobel gradient magnitude, thresholded to the top percentile of
    pixels -- a simple, standard classical edge map, no learning."""
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, thresh_percentile)
    ys, xs = np.nonzero(mag > thresh)
    return np.stack([xs, ys], -1).astype(np.float64)


def line_theta_r(p1, p2, size):
    """Two points -> (theta, r) in the same convention hough.py's 'line'
    family and tusimple_dataset.py's letterbox_params use: origin at
    image center, theta in [0, pi)."""
    cx = cy = (size - 1) / 2.0
    d = p2 - p1
    if np.allclose(d, 0):
        return None
    # normal to the line direction d
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-9)
    theta = np.arctan2(n[1], n[0]) % np.pi
    if np.arctan2(n[1], n[0]) < 0 and not np.isclose(theta, 0):
        sign = -1
    else:
        sign = 1
    mid = (p1 + p2) / 2.0 - np.array([cx, cy])
    r = sign * float(np.dot(mid, n))
    return theta, r


def point_line_distance(pts, theta, r, size):
    cx = cy = (size - 1) / 2.0
    nx, ny = np.cos(theta), np.sin(theta)
    return np.abs((pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny - r)


def ransac_lines(pts, size, topn=4, iters=200, tol=2.5, min_inliers=15, seed=0):
    if len(pts) < 2:
        return []
    rng = np.random.default_rng(seed)
    remaining = pts.copy()
    dets = []
    for _ in range(topn):
        if len(remaining) < min_inliers:
            break
        best_inliers, best_params, best_count = None, None, 0
        for _ in range(iters):
            i, j = rng.choice(len(remaining), 2, replace=False)
            fit = line_theta_r(remaining[i], remaining[j], size)
            if fit is None:
                continue
            theta, r = fit
            d = point_line_distance(remaining, theta, r, size)
            inliers = d < tol
            count = inliers.sum()
            if count > best_count:
                best_count, best_inliers, best_params = count, inliers, (theta, r)
        if best_params is None or best_count < min_inliers:
            break
        theta, r = best_params
        score = min(1.0, best_count / 200.0)  # crude confidence proxy
        dets.append((theta, r, score))
        remaining = remaining[~best_inliers]
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--edge-percentile", type=float, default=90)
    ap.add_argument("--tol", type=float, default=2.5)
    args = ap.parse_args()

    full = RealTuSimpleDataset(args.root, "line", size=args.size,
                               max_images=args.max_images)
    n = len(full)
    print(f"loaded {n} real frames with >=1 'line' lane")
    n_test = max(1, n // 10)
    idx = list(range(n))
    rng = np.random.default_rng(0)   # SAME seed/shuffle as train_real.py
    rng.shuffle(idx)
    test_idx = idx[:n_test]           # first n_test after shuffle = test set
    print(f"test set: {len(test_idx)} frames (same split train_real.py used)")

    dets_all, gts_all = [], []
    for k, i in enumerate(test_idx):
        img_t, params, count = full[i]
        img = img_t[0].numpy()
        gts = [params[j].numpy()[:2] for j in range(count)]
        pts = edge_pixels(img, args.edge_percentile)
        dets = ransac_lines(pts, args.size, tol=args.tol)
        dets_all.append(dets)
        gts_all.append(gts)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(test_idx)} processed")

    n_det = sum(len(d) for d in dets_all)
    n_gt = sum(len(g) for g in gts_all)
    p, r, f = prf(dets_all, gts_all, "line", args.size)
    print(f"\n[RANSAC line baseline] detections {n_det} vs gt {n_gt}  "
          f"P {p:.3f} R {r:.3f} F {f:.3f}")


if __name__ == "__main__":
    main()
