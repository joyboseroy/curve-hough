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
    ap.add_argument("--family", default="parabola",
                    choices=["parabola", "circle", "ellipse"])
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
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--resume", default="",
                    help="load weights and continue training for --epochs more "
                         "epochs (unlike --load, which skips straight to eval)")
    ap.add_argument("--sweep", action="store_true",
                    help="load a checkpoint and report P/R/F across thresholds, no training")
    ap.add_argument("--topk", type=int, default=None,
                    help="max detections per image (defaults to model's, 4)")
    ap.add_argument("--diag", action="store_true",
                    help="print matched curve-EA similarity distribution")
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
                          anchor_bins=24 if args.smoke else 32,
                          topk=args.topk if args.topk else 4).to(dev)

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
                torch.save(model.state_dict(), args.save)  # checkpoint every
                print(f"  checkpoint saved to {args.save}")  # epoch, not just at the end
        if args.save:
            print(f"training complete, final weights in {args.save}")

    # evaluation
    model.eval()
    gts_by_img = []
    for i in range(len(test)):
        _, params, k = test[i]
        gts_by_img.append([params[j].numpy() for j in range(k)])

    def run_eval(thresh):
        dets_all = []
        for i in range(len(test)):
            img, _, _ = test[i]
            dets_all.append(model.detect(img[None].to(dev), thresh=thresh)[0])
        n_det = sum(len(d) for d in dets_all)
        n_gt = sum(len(g) for g in gts_by_img)
        (p, r, f), sims = prf(dets_all, gts_by_img, args.family, args.size,
                              return_sims=True)
        print(f"thresh {thresh:.2f}: detections {n_det} vs gt {n_gt}  "
              f"P {p:.3f} R {r:.3f} F {f:.3f}")
        if args.diag and len(sims):
            q = np.percentile(sims, [10, 25, 50, 75, 90])
            print(f"  matched curve-EA similarity: mean {sims.mean():.3f}  "
                  f"p10/p25/p50/p75/p90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}/"
                  f"{q[3]:.2f}/{q[4]:.2f}  (n={len(sims)})")
        return f

    if args.sweep:
        for t in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            run_eval(t)
    else:
        run_eval(args.thresh)


if __name__ == "__main__":
    main()
