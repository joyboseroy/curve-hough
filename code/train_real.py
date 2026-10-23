"""Train on real TuSimple images (letterboxed), one model per family.

Usage:
    python train_real.py --root /content/tusimple/TUSimple/train_set \
        --family lane --epochs 8 --save lane_real.pt --sweep
    python train_real.py --root /content/tusimple/TUSimple/train_set \
        --family line --epochs 8 --save line_real.pt --sweep

Splits the loaded frames 80/10/10 train/val/test with a seeded shuffle (see
real_data/split.py). The detection threshold is selected on the validation
frames and the test frames are scored once, the same protocol as train.py.
No official held-out TuSimple test labels are used.

Per-seed run with results saved:
    python train_real.py --root ... --family lane --pool max --seed 1 \
        --epochs 8 --json-out lane_seed1.json
"""
import argparse
import json
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "real_data"))
from hough import FactorizedDHT, stage1_loss
from real_data.tusimple_dataset import RealTuSimpleDataset
from real_data.split import split_indices
import train as T


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    counts = [b[2] for b in batch]
    return imgs, params, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="TuSimple train_set dir")
    ap.add_argument("--family", required=True, choices=["lane", "line"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap total labeled frames loaded, for a fast first pass")
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresh", type=float, default=None,
                    help="fixed score threshold; default: select on the validation frames")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the split, initialization and data order")
    ap.add_argument("--pool", default="max", choices=["logsumexp", "mean", "max"],
                    help="Stage-1 pooling over probe shapes (lane only; the "
                         "synthetic results use max)")
    ap.add_argument("--probe-shapes", type=int, default=6)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--diag", action="store_true",
                    help="print matched curve-EA similarity distribution")
    ap.add_argument("--vertex-margin", type=float, default=0.5,
                    help="lane only: drop lanes whose transformed vertex falls "
                         "more than this fraction of the image size outside the "
                         "frame (0.5 default; try smaller to keep only well-"
                         "in-frame vertices, larger to keep more marginal ones)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    is_line = (args.family == "line")

    full = RealTuSimpleDataset(args.root, args.family, size=args.size,
                               max_images=args.max_images,
                               vertex_margin=args.vertex_margin)
    n = len(full)
    print(f"loaded {n} real frames with >=1 '{args.family}' lane")
    train_idx, val_idx, test_idx = split_indices(n, args.seed)
    train = torch.utils.data.Subset(full, train_idx)
    val = torch.utils.data.Subset(full, val_idx)
    test = torch.utils.data.Subset(full, test_idx)
    print(f"train {len(train)} / val {len(val)} / test {len(test)}")
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)

    model_kwargs = dict(topk=4, pool=args.pool)
    if is_line:
        model_kwargs["theta_bins"] = 60
        model_kwargs["r_bins"] = 60
    else:
        model_kwargs["anchor_bins"] = 32
        model_kwargs["probe_shapes"] = args.probe_shapes
        # The default shape range (0.004, 0.06) was tuned on the synthetic
        # benchmark and doesn't necessarily cover real curvature -- 24.4% of
        # real TuSimple lane 'a' values fell outside it in an earlier check.
        # Compute from this dataset's actual distribution instead: p5-p95
        # covers the bulk of real values while not letting a few extreme
        # outliers blow the range out (which would waste probe/dense
        # resolution on rarely-seen curvatures).
        all_a = []
        for i in range(len(full)):
            _, params, k = full[i]
            for j in range(k):
                a_val = abs(float(params[j][2]))
                if a_val > 1e-6:  # skip exact zeros (shouldn't occur for
                    all_a.append(a_val)  # 'lane', which excludes near-straight
        if all_a:
            lo = max(1e-4, np.percentile(all_a, 5))
            hi = np.percentile(all_a, 95)
            print(f"shape range from data (p5-p95 of |a|, n={len(all_a)}): "
                  f"[{lo:.5f}, {hi:.5f}]  (default was [0.004, 0.06])")
            model_kwargs["shape_range"] = (lo, hi)
        else:
            print("no curvature values found -- keeping default shape_range")
    model = FactorizedDHT(args.family, args.size, **model_kwargs).to(dev)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=dev))
        print(f"loaded {args.load}")
    else:
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=dev))
            print(f"resumed from {args.resume}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            model.train()
            losses = []
            for imgs, params, counts in tl:
                imgs = imgs.to(dev)
                P1, P2, picked, feat = model(imgs)
                if is_line:
                    l1 = model.stage1_loss(P1, params, counts)
                    l2 = torch.tensor(0.0, device=dev)
                else:
                    l1 = stage1_loss(P1, params, counts, args.size)
                    l2 = model.stage2_loss(feat, params, counts)
                loss = l1 + l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append((l1.item(), l2.item()))
            m = np.mean(losses, axis=0)
            print(f"epoch {ep}: stage1 {m[0]:.4f}  stage2 {m[1]:.4f}")
            if args.save:
                torch.save(model.state_dict(), args.save)
                print(f"  checkpoint saved to {args.save}")

    model.eval()
    result = T.evaluate(model, args, val, test, dev)
    result["split"] = dict(n_train=len(train), n_val=len(val), n_test=len(test),
                           seed=args.seed)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=1, default=float)
        print(f"results written to {args.json_out}")


if __name__ == "__main__":
    main()
