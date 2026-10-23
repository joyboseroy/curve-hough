"""RANSAC baselines on real TuSimple images, scored like the learned model.

Same frames, same split (real_data/split.py), same metric as train_real.py.
Classical edges (Sobel) plus RANSAC, no learning. Two families:
  line : two-point RANSAC lines (theta, r)
  lane : three-point RANSAC parabolas x = a (y - y0)^2 + x0, fitted by
         transposing the edge map and reusing baselines.ransac

Each candidate carries its inlier count as score. Like the learned model, the
rule-based baseline gets a minimum-score threshold selected on the validation
frames, then the test frames are scored once. The untuned variant (always keep
the top 4 candidates) is printed for reference.

Usage:
    python real_data/ransac_real_baseline.py --root /content/tusimple/TUSimple/train_set \
        --family line --seed 0 --json-out ransac_line_seed0.json
"""
import argparse
import json
import os
import sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from metrics import prf
from baselines import ransac as ransac_conic
from tusimple_dataset import RealTuSimpleDataset
from split import split_indices
from ransac_line_baseline import edge_pixels, line_theta_r, point_line_distance

COUNT_GRID = [10, 15, 20, 30, 40, 60, 80, 100, 150, 200]


def candidates(img, family, size, edge_percentile, tol, topn):
    """Detections as (params..., inlier_count)."""
    if family == "line":
        pts = edge_pixels(img, edge_percentile)
        res = []
        rng_seed = 0
        remaining = pts.copy()
        # re-run ransac_lines but keep raw inlier counts as the score
        rng = np.random.default_rng(rng_seed)
        for _ in range(topn):
            if len(remaining) < 10:
                break
            best_count, best_in, best_par = 0, None, None
            for _ in range(200):
                i, j = rng.choice(len(remaining), 2, replace=False)
                fit = line_theta_r(remaining[i], remaining[j], size)
                if fit is None:
                    continue
                d = point_line_distance(remaining, fit[0], fit[1], size)
                inl = d < tol
                if inl.sum() > best_count:
                    best_count, best_in, best_par = int(inl.sum()), inl, fit
            if best_par is None or best_count < 10:
                break
            res.append((best_par[0], best_par[1], float(best_count)))
            remaining = remaining[~best_in]
        return res
    # lane: transpose the binary edge map so x <-> y, fit y' = c (x'-p)^2 + q
    pts = edge_pixels(img, edge_percentile)
    edge_img = np.zeros((size, size), np.float32)
    edge_img[pts[:, 1].astype(int), pts[:, 0].astype(int)] = 1.0
    out = []
    for xp, yp, c, count in ransac_conic(edge_img.T.copy(), "parabola", size,
                                         tol=tol, topn=topn):
        out.append((yp, xp, c, count))          # lane params (x0, y0, a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", default="line", choices=["line", "lane"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edge-percentile", type=float, default=90)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--vertex-margin", type=float, default=0.5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    full = RealTuSimpleDataset(args.root, args.family, size=args.size,
                               max_images=args.max_images,
                               vertex_margin=args.vertex_margin)
    n = len(full)
    _, val_idx, test_idx = split_indices(n, args.seed)
    print(f"loaded {n} frames; val {len(val_idx)} / test {len(test_idx)} "
          f"(same split as train_real.py, seed {args.seed})")

    def run(idx):
        dets_all, gts_all = [], []
        for k, i in enumerate(idx):
            img_t, params, count = full[i]
            dets_all.append(candidates(img_t[0].numpy(), args.family, args.size,
                                       args.edge_percentile, args.tol, args.topn))
            gts_all.append([params[j].numpy() for j in range(count)])
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(idx)}")
        return dets_all, gts_all

    def score(dets, gts, min_count):
        filt = [[d for d in per if d[-1] >= min_count] for per in dets]
        p, r, f = prf(filt, gts, args.family, args.size)
        return dict(min_count=min_count, n_det=sum(len(d) for d in filt),
                    n_gt=sum(len(g) for g in gts), P=p, R=r, F=f)

    print("validation frames:")
    vd, vg = run(val_idx)
    val_sweep = [score(vd, vg, c) for c in COUNT_GRID]
    best = max(val_sweep, key=lambda r: r["F"])
    print("test frames:")
    td, tg = run(test_idx)
    tuned = score(td, tg, best["min_count"])
    untuned = score(td, tg, 0)
    oracle = max((score(td, tg, c) for c in COUNT_GRID), key=lambda r: r["F"])
    print(f"min inlier count {best['min_count']} (selected on validation)")
    print(f"[RANSAC {args.family}, tuned]   TEST dets {tuned['n_det']} vs gt "
          f"{tuned['n_gt']}  P {tuned['P']:.3f} R {tuned['R']:.3f} F {tuned['F']:.3f}")
    print(f"[RANSAC {args.family}, untuned top-{args.topn}] TEST dets {untuned['n_det']}  "
          f"P {untuned['P']:.3f} R {untuned['R']:.3f} F {untuned['F']:.3f}")
    print(f"test-sweep oracle F {oracle['F']:.3f} (gap vs tuned {oracle['F'] - tuned['F']:+.3f})")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(args=vars(args), chosen_min_count=best["min_count"],
                           tuned=tuned, untuned=untuned, test_sweep_oracle=oracle,
                           val_sweep=val_sweep,
                           split=dict(n_val=len(val_idx), n_test=len(test_idx),
                                      seed=args.seed)), fh, indent=1, default=float)
        print(f"results written to {args.json_out}")


if __name__ == "__main__":
    main()
