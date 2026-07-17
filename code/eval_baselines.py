"""Evaluate baselines on the same synthetic benchmark and curve-EA metric.

Rule-based (no training): classical Hough, RANSAC.
Learned (trained here): regression, query (DETR-lite).

Usage:
    python eval_baselines.py --model classical --family parabola --n-test 50
    python eval_baselines.py --model ransac --family parabola --n-test 50
    python eval_baselines.py --model regression --family parabola --epochs 10 --n-train 1000
    python eval_baselines.py --model query --family parabola --epochs 10 --n-train 1000
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticCurves
from baselines import classical_hough, ransac, RegressionHead, QueryHead, hungarian_param_loss
from metrics import prf


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    counts = [b[2] for b in batch]
    return imgs, params, counts


def learned_dets(pred, logits, thresh):
    """pred: [K,3], logits: [K] -> list of (x0,y0,s,score) above thresh."""
    probs = torch.sigmoid(logits)
    dets = []
    for k in range(pred.shape[0]):
        if probs[k] > thresh:
            p = pred[k].tolist()
            dets.append((p[0], p[1], p[2], float(probs[k])))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["classical", "ransac", "regression", "query"])
    ap.add_argument("--family", default="parabola", choices=["parabola", "circle"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--easy", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.5)
    # classical/RANSAC are O(anchor^2 * shapes) per image; keep test set small
    # or lower these for speed.
    ap.add_argument("--classical-anchor-bins", type=int, default=16)
    ap.add_argument("--classical-dense-shapes", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    kw = dict(noise=0.0, occlude=False, distractors=False, max_curves=1) \
        if args.easy else {}
    test = SyntheticCurves(args.family, args.size, args.n_test, seed=2, **kw)
    gts_by_img = [[test[i][1][j].numpy() for j in range(test[i][2])]
                  for i in range(len(test))]

    if args.model in ("classical", "ransac"):
        dets_all = []
        for i in range(len(test)):
            img = test[i][0][0].numpy()
            if args.model == "classical":
                dets_all.append(classical_hough(
                    img, args.family, args.size,
                    anchor_bins=args.classical_anchor_bins,
                    dense_shapes=args.classical_dense_shapes, topn=4))
            else:
                dets_all.append(ransac(img, args.family, args.size, topn=4))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test)} images processed")
    else:
        train = SyntheticCurves(args.family, args.size, args.n_train, seed=1, **kw)
        tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
        model = (RegressionHead() if args.model == "regression" else QueryHead()).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            model.train()
            losses = []
            for imgs, params, counts in tl:
                imgs, params = imgs.to(dev), params.to(dev)
                pred, logits = model(imgs)
                loss = hungarian_param_loss(pred, logits, params, counts)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"epoch {ep}: loss {np.mean(losses):.4f}")

        model.eval()
        dets_all = []
        with torch.no_grad():
            for i in range(len(test)):
                img = test[i][0][None].to(dev)
                pred, logits = model(img)
                dets_all.append(learned_dets(pred[0].cpu(), logits[0].cpu(), args.thresh))

    n_det = sum(len(d) for d in dets_all)
    n_gt = sum(len(g) for g in gts_by_img)
    p, r, f = prf(dets_all, gts_by_img, args.family, args.size)
    print(f"[{args.model}] detections {n_det} vs gt {n_gt}  P {p:.3f} R {r:.3f} F {f:.3f}")


if __name__ == "__main__":
    main()
