"""Data-driven figures from a REAL trained checkpoint -- run this on your
machine against your actual .pt files. Produces three PNGs per run:
  <out>/stage1_heatmap_<family>.png   -- Stage-1 accumulator + GT + peaks
  <out>/stage2_profile_<family>.png   -- Stage-2 shape profile for one peak
  <out>/qualitative_<family>.png      -- image with GT (green) vs pred (red)

Usage:
    python make_figures.py --family parabola --load parabola_hard.pt --idx 3
    python make_figures.py --family ellipse --load ellipse_hard.pt --idx 0
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from hough import FactorizedDHT, stage1_accumulate
from dataset import SyntheticCurves
from metrics import sample_curve

NAVY = "#1E2761"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["parabola", "circle", "ellipse"])
    ap.add_argument("--load", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--anchor-bins", type=int, default=32)
    ap.add_argument("--easy", action="store_true")
    ap.add_argument("--idx", type=int, default=0, help="which test image to visualize")
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--out", default=".",
                    help="output dir for the PNGs (default: current directory, "
                         "same place as your .pt checkpoint files)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kw = dict(noise=0.0, occlude=False, distractors=False, max_curves=1) \
        if args.easy else {}
    test = SyntheticCurves(args.family, args.size, args.idx + 1, seed=2, **kw)
    img, params, k = test[args.idx]
    gts = [params[j].numpy() for j in range(k)]

    model = FactorizedDHT(args.family, args.size, anchor_bins=args.anchor_bins)
    model.load_state_dict(torch.load(args.load, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X = model.encoder(img[None])
        feat = X.flatten(2)
        Y1 = stage1_accumulate(feat, model.bank1, model.Ba)
        P1 = model.head1(Y1)
        prob_map = torch.sigmoid(P1)[0, 0].numpy()      # [Ba, Ba]
        dets = model.detect(img[None], thresh=args.thresh)[0]

    scale = (args.size - 1) / (model.Ba - 1)

    # ---------------- Figure 1: Stage-1 accumulator heatmap ----------------
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(prob_map, origin="lower", cmap="Blues", vmin=0, vmax=1,
                   extent=[0, args.size, 0, args.size])
    for g in gts:
        ax.plot(g[0], g[1], marker="*", color="#E63946", markersize=16,
                markeredgecolor="white", markeredgewidth=0.8, label="ground truth")
    for d in dets:
        ax.plot(d[0], d[1], marker="o", markerfacecolor="none",
                markeredgecolor=NAVY, markeredgewidth=1.8, markersize=13,
                label="detected peak")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Stage-1 anchor accumulator ({args.family})", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(anchor)")
    plt.tight_layout()
    p1 = f"{args.out}/stage1_heatmap_{args.family}.png"
    plt.savefig(p1, dpi=200); plt.close()

    # ---------------- Figure 2: Stage-2 shape profile ----------------------
    if dets:
        # recompute stage-2 profile for the top-scoring detection's anchor
        best = max(dets, key=lambda d: d[-1])
        # find nearest anchor bin to the detected (ax, ay)
        ax_scaled = round(best[0] / scale)
        ay_scaled = round(best[1] / scale)
        a_idx = int(ay_scaled) * model.Ba + int(ax_scaled)
        a_idx = max(0, min(a_idx, len(model.anchors) - 1))
        with torch.no_grad():
            bank2 = model.bank2_for(a_idx, "cpu")
            from hough import vote
            y2 = vote(feat, bank2)
            s_prof = torch.sigmoid(model.head2(y2)[0, 0]).numpy()  # [S2]

        fig, ax2 = plt.subplots(figsize=(6.0, 3.2))
        ax2.plot(s_prof, color=NAVY, linewidth=1.3)
        ax2.fill_between(range(len(s_prof)), s_prof, color=NAVY, alpha=0.12)
        if gts:
            gt_shape = gts[0][2:2 + model.shape_dim]
            gt_norm = gt_shape / model._dense_range
            dense_norm = model._dense_arr / model._dense_range
            s_bin = int(np.argmin(((dense_norm - gt_norm) ** 2).sum(-1)))
            ax2.axvline(s_bin, color="#E63946", linestyle="--", label="ground-truth shape bin")
        pred_bin = int(np.argmax(s_prof))
        ax2.axvline(pred_bin, color=NAVY, linestyle=":", label="predicted peak")
        ax2.set_xlabel("shape hypothesis index (flattened)")
        ax2.set_ylabel("P(shape)")
        ax2.set_title(f"Stage-2 shape profile at top peak ({args.family})", fontsize=11)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        p2 = f"{args.out}/stage2_profile_{args.family}.png"
        plt.savefig(p2, dpi=200); plt.close()
    else:
        p2 = None
        print("no detections above threshold -- skipping stage-2 figure; try a lower --thresh")

    # ---------------- Figure 3: qualitative detections ----------------------
    fig, ax3 = plt.subplots(figsize=(4.6, 4.6))
    ax3.imshow(img[0].numpy(), cmap="gray", origin="upper", vmin=0, vmax=1)
    for g in gts:
        pts, _ = sample_curve(args.family, g, args.size, n=300)
        if len(pts):
            ax3.plot(pts[:, 0], pts[:, 1], color="#2ECC71", linewidth=2.2, label="ground truth")
    for d in dets:
        pts, _ = sample_curve(args.family, d[:-1], args.size, n=300)
        if len(pts):
            ax3.plot(pts[:, 0], pts[:, 1], color="#E63946", linewidth=1.6,
                     linestyle="--", label="detected")
    handles, labels = ax3.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax3.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, framealpha=0.9)
    ax3.set_title(f"Qualitative detection ({args.family}, idx={args.idx})", fontsize=11)
    ax3.set_xlim(0, args.size); ax3.set_ylim(args.size, 0)
    ax3.axis("off")
    plt.tight_layout()
    p3 = f"{args.out}/qualitative_{args.family}.png"
    plt.savefig(p3, dpi=200); plt.close()

    print(f"saved: {p1}")
    if p2:
        print(f"saved: {p2}")
    print(f"saved: {p3}")


if __name__ == "__main__":
    main()
