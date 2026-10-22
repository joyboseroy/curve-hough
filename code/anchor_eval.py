"""Stage-1 diagnostics for the anchor-first factorization.

Two measurements, both aimed at the same structural weakness: Stage 2 is only
run around the top-k Stage-1 anchors, so a ground-truth anchor that is not in
the retained set can never be recovered.

1. Top-k anchor recall: for each k, the fraction of ground-truth curves whose
   anchor is matched (one-to-one, within `tol` anchor bins) by one of the k
   highest Stage-1 peaks. This is a strict localization measure, NOT a bound
   on end-to-end recall: the F/R metric averages over lenient similarity
   thresholds, so a detection whose anchor is a few bins off can still match
   its curve. The oracle-anchor ceiling below is the clean bound.
2. Separation sweep: two curves whose anchors are `sep` pixels apart (0 = the
   same anchor). Reports anchor recall at the model's k and detection P/R/F.
   Anchors closer than the 3x3 peak window collapse into one Stage-1 peak, so
   this quantifies how much that costs.

Usage:
    python anchor_eval.py --family parabola --size 64 --load model.pt \
        --thresh-from run.json --json-out anchors.json
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import train as T
from dataset import SyntheticCurves, SeparationCurves


def stage1_peaks(model, img, k_max):
    """Top-k_max Stage-1 peaks for one image, same peak rule as detect():
    3x3 local maxima of the logit map, ranked by sigmoid score.
    Returns list of ((row, col), score), best first."""
    P1 = model.stage1_logits(img)
    pooled = F.max_pool2d(P1, 3, stride=1, padding=1)
    peaks = ((P1 == pooled).float() * torch.sigmoid(P1))[0, 0].flatten()
    vals, idxs = peaks.topk(min(k_max, peaks.numel()))
    Ba = model.Ba
    return [((i // Ba, i % Ba), v) for i, v in zip(idxs.tolist(), vals.tolist())]


def gt_bins(model, gts):
    out = []
    for g in gts:
        idx = model.anchor_index(g[0], g[1])
        out.append((idx // model.Ba, idx % model.Ba))
    return out


def matched_count(peak_bins, gt_bin_list, tol):
    """One-to-one count of ground-truth anchors matched by a peak within `tol`
    bins (Chebyshev). One-to-one matters: with two nearby ground truths a single
    peak must not be credited for both."""
    if not peak_bins or not gt_bin_list:
        return 0
    D = np.array([[max(abs(p[0] - g[0]), abs(p[1] - g[1])) for g in gt_bin_list]
                  for p in peak_bins], dtype=float)
    cost = np.where(D <= tol, D, 1e6)
    ri, ci = linear_sum_assignment(cost)
    return int(sum(D[r, c] <= tol for r, c in zip(ri, ci)))


@torch.no_grad()
def topk_anchor_recall(model, ds, dev, ks=(1, 2, 3, 4, 6, 8), tols=(0, 1, 2)):
    model.eval()
    k_max = max(ks)
    hits = {(t, k): 0 for t in tols for k in ks}
    n_gt = 0
    for i in range(len(ds)):
        img, params, cnt = ds[i]
        gts = [params[j].numpy() for j in range(cnt)]
        gb = gt_bins(model, gts)
        peaks = [p for p, _ in stage1_peaks(model, img[None].to(dev), k_max)]
        n_gt += len(gb)
        for t in tols:
            for k in ks:
                hits[(t, k)] += matched_count(peaks[:k], gb, t)
    return {t: {k: hits[(t, k)] / max(n_gt, 1) for k in ks} for t in tols}, n_gt


@torch.no_grad()
def separation_sweep(model, args, dev, thresh, seps=(0, 2, 4, 6, 8, 12, 16, 24), n=200,
                     tol=1, seed=7):
    model.eval()
    rows = []
    for sep in seps:
        ds = SeparationCurves(args.family, args.size, n, seed=seed, sep=sep,
                              noise=0.0 if args.easy else args.noise)
        hit, n_gt = 0, 0
        for i in range(len(ds)):
            img, params, cnt = ds[i]
            gb = gt_bins(model, [params[j].numpy() for j in range(cnt)])
            peaks = [p for p, _ in stage1_peaks(model, img[None].to(dev), model.topk)]
            hit += matched_count(peaks, gb, tol)
            n_gt += len(gb)
        dets, gts = T.all_detections(model, ds, dev)
        r = T.score_at(dets, gts, args.family, args.size, thresh)
        rows.append(dict(sep=sep, anchor_recall=hit / max(n_gt, 1),
                         P=r["P"], R=r["R"], F=r["F"], n_det=r["n_det"], n_gt=r["n_gt"]))
    return rows


@torch.no_grad()
def oracle_anchor_eval(model, ds, dev, family, size):
    """Ceiling for the cascade: Stage 2 run at the ground-truth anchors, so
    Stage-1 errors cannot occur. Compared with the real F, the difference is
    the cost of Stage 1 (localization and missed anchors)."""
    model.eval()
    dets, gts = [], []
    for i in range(len(ds)):
        img, params, cnt = ds[i]
        g = [params[j].numpy() for j in range(cnt)]
        dets.append(model.detect_at_anchors(img[None].to(dev), [(p[0], p[1]) for p in g]))
        gts.append(g)
    return T.score_at(dets, gts, family, size, 0.0)


def main():
    ap = T.build_parser()
    ap.add_argument("--thresh-from", default="",
                    help="JSON from train.py --json-out; uses its validation-chosen "
                         "threshold for the separation sweep")
    ap.add_argument("--seps", default="0,2,4,6,8,12,16,24",
                    help="comma-separated anchor separations in pixels")
    ap.add_argument("--n-sep", type=int, default=200, help="images per separation")
    args = ap.parse_args()
    if args.family == "line":
        raise SystemExit("line has no anchor/Stage-2 split; nothing to measure")
    if args.smoke:
        args.size, args.n_test, args.n_sep = 64, 8, 8
        args.seps = "0,8"
    if not args.load:
        raise SystemExit("--load a trained checkpoint (train.py --save ...)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = T.build_model(args, dev)
    model.load_state_dict(torch.load(args.load, map_location=dev))
    model.eval()

    thresh = args.thresh
    if thresh is None and args.thresh_from:
        thresh = json.load(open(args.thresh_from))["chosen_thresh"]
    if thresh is None:
        thresh = 0.25
        print("no --thresh/--thresh-from given; using 0.25 for the separation sweep")

    kw = (dict(noise=0.0, occlude=False, distractors=False, max_curves=1) if args.easy
          else dict(noise=args.noise, occ_max=args.occ_max))
    test = SyntheticCurves(args.family, args.size, args.n_test, seed=2 + 10 * args.seed, **kw)
    recall, n_gt = topk_anchor_recall(model, test, dev)
    print(f"top-k Stage-1 anchor recall on the test distribution (n_gt={n_gt}); "
          f"model keeps k={model.topk}")
    ks = sorted(next(iter(recall.values())).keys())
    print("  tol(bins) | " + " ".join(f"k={k:<3d}" for k in ks))
    for t, row in recall.items():
        print(f"  {t:^9d} | " + " ".join(f"{row[k]:5.3f}" for k in ks))

    orc = oracle_anchor_eval(model, test, dev, args.family, args.size)
    real = T.score_at(*T.all_detections(model, test, dev), args.family, args.size, thresh)
    print(f"oracle-anchor ceiling (Stage 2 at ground-truth anchors, no Stage 1): "
          f"P {orc['P']:.3f} R {orc['R']:.3f} F {orc['F']:.3f}   "
          f"vs real pipeline at thresh {thresh:.2f}: F {real['F']:.3f}   "
          f"(cost of Stage 1: {orc['F'] - real['F']:+.3f})")

    seps = [float(s) for s in args.seps.split(",")]
    rows = separation_sweep(model, args, dev, thresh, seps=seps, n=args.n_sep)
    print(f"separation sweep (two curves, anchors `sep` px apart, thresh {thresh:.2f}):")
    print("  sep(px) anchor_recall@k    P      R      F")
    for r in rows:
        print(f"  {r['sep']:6.1f}   {r['anchor_recall']:10.3f}   {r['P']:.3f}  "
              f"{r['R']:.3f}  {r['F']:.3f}")

    if args.json_out:
        out = dict(args=vars(args), oracle_anchor=orc, real_pipeline=real, topk_recall={str(t): {str(k): v for k, v in row.items()}
                                                 for t, row in recall.items()},
                   separation=rows, thresh=thresh)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"results written to {args.json_out}")


if __name__ == "__main__":
    main()
