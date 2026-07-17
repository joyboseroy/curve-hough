"""Train the factorized DHT on synthetic curves and evaluate with curve-EA.

Usage:
    python train.py --family parabola --epochs 5 --n-train 2000
Smoke test:
    python train.py --smoke
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticCurves
from hough import FactorizedDHT, stage1_loss
from metrics import prf


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    counts = [b[2] for b in batch]
    return imgs, params, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="parabola", choices=["parabola", "circle"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--easy", action="store_true",
                    help="no noise/occlusion/distractors, single curve")
    ap.add_argument("--thresh", type=float, default=0.25)
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.n_train, args.n_test, args.size = 1, 16, 8, 64

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    kw = dict(noise=0.0, occlude=False, distractors=False, max_curves=1) \
        if args.easy else {}
    train = SyntheticCurves(args.family, args.size, args.n_train, seed=1, **kw)
    test = SyntheticCurves(args.family, args.size, args.n_test, seed=2, **kw)
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    model = FactorizedDHT(args.family, args.size,
                          anchor_bins=24 if args.smoke else 32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        losses = []
        for imgs, params, counts in tl:
            imgs = imgs.to(dev)
            P1, P2, picked, feat = model(imgs)
            l1 = stage1_loss(P1, params, counts, args.size)
            l2 = model.stage2_loss(feat, params, counts)
            loss = l1 + l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append((l1.item(), l2.item()))
        m = np.mean(losses, axis=0)
        print(f"epoch {ep}: stage1 {m[0]:.4f}  stage2 {m[1]:.4f}")

    # evaluation
    model.eval()
    dets_all, gts_all = [], []
    for i in range(len(test)):
        img, params, k = test[i]
        dets = model.detect(img[None].to(dev), thresh=args.thresh)[0]
        dets_all.append(dets)
        gts_all.append([params[j].numpy() for j in range(k)])
    n_det = sum(len(d) for d in dets_all)
    n_gt = sum(len(g) for g in gts_all)
    print(f"detections {n_det} vs ground truth {n_gt}")
    p, r, f = prf(dets_all, gts_all, args.family, args.size)
    print(f"curve-EA avg P {p:.3f} R {r:.3f} F {f:.3f}")


if __name__ == "__main__":
    main()
