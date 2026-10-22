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
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticCurves
from baselines import (classical_hough, ransac, RegressionHead, QueryHead,
                       hungarian_param_loss, FAMILY_NPARAMS)
from metrics import prf


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    counts = [b[2] for b in batch]
    return imgs, params, counts


def learned_dets(pred, logits, thresh):
    """pred: [K,n_params], logits: [K] -> list of (params..., score) above thresh."""
    probs = torch.sigmoid(logits)
    dets = []
    for k in range(pred.shape[0]):
        if probs[k] > thresh:
            dets.append(tuple(pred[k].tolist()) + (float(probs[k]),))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["classical", "ransac", "regression", "query"])
    ap.add_argument("--family", default="parabola", choices=["parabola", "circle", "ellipse"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--easy", action="store_true")
    ap.add_argument("--n-val", type=int, default=200,
                    help="validation images used ONLY to pick the threshold (learned models)")
    ap.add_argument("--seed", type=int, default=0,
                    help="init/shuffle seed AND data seed offset (train/test/val = 1/2/3 + 10*seed)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="fixed threshold; if omitted it is selected on the validation set")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--tune-score", action="store_true",
                    help="rule-based models only: also select a minimum detection score "
                         "(RANSAC inlier count, Hough accumulator value) on the validation "
                         "set and score the test set once, the same treatment the learned "
                         "models and the factorized DHT get")
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=4.0)
    # classical/RANSAC are O(anchor^2 * shapes) per image; keep test set small
    # or lower these for speed.
    ap.add_argument("--classical-anchor-bins", type=int, default=16)
    ap.add_argument("--classical-dense-shapes", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base = 10 * args.seed
    kw = dict(noise=0.0, occlude=False, distractors=False, max_curves=1) \
        if args.easy else {}
    test = SyntheticCurves(args.family, args.size, args.n_test, seed=2 + base, **kw)
    gts_by_img = [[test[i][1][j].numpy() for j in range(test[i][2])]
                  for i in range(len(test))]

    if args.model == "classical" and args.family == "ellipse":
        raise SystemExit("classical dense Hough is not run for ellipse: a dense "
                         "5-D accumulator is the intractable case the paper is about "
                         "(see measure_memory.py). Use --model ransac.")
    def rule_dets(ds):
        out = []
        for i in range(len(ds)):
            img = ds[i][0][0].numpy()
            if args.model == "classical":
                out.append(classical_hough(
                    img, args.family, args.size,
                    anchor_bins=args.classical_anchor_bins,
                    dense_shapes=args.classical_dense_shapes, topn=4))
            else:
                out.append(ransac(img, args.family, args.size, topn=4))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(ds)} images processed")
        return out

    if args.model in ("classical", "ransac"):
        dets_all = rule_dets(test)
    else:
        npar = FAMILY_NPARAMS[args.family]
        model = (RegressionHead(n_params=npar) if args.model == "regression"
                 else QueryHead(n_params=npar)).to(dev)
        if args.load:
            model.load_state_dict(torch.load(args.load, map_location=dev))
            print(f"loaded {args.load}")
        else:
            train = SyntheticCurves(args.family, args.size, args.n_train, seed=1 + base, **kw)
            tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            for ep in range(args.epochs):
                model.train()
                losses = []
                for imgs, params, counts in tl:
                    imgs, params = imgs.to(dev), params.to(dev)
                    pred, logits = model(imgs)
                    loss = hungarian_param_loss(pred, logits, params, counts,
                                                family=args.family,
                                                pos_weight=args.pos_weight)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                print(f"epoch {ep}: loss {np.mean(losses):.4f}")
            if args.save:
                torch.save(model.state_dict(), args.save)
                print(f"saved {args.save}")

        # Protocol (v24): predictions are cached once per split; the score
        # threshold is selected on a held-out VALIDATION set and the test set
        # is scored once at that threshold. The old test-sweep best-F is still
        # printed as an "oracle" so the optimism of the old protocol is visible.
        model.eval()
        val = SyntheticCurves(args.family, args.size, args.n_val, seed=3 + base, **kw)

        def raw_preds(ds):
            raw = []
            with torch.no_grad():
                for i in range(len(ds)):
                    pred, logits = model(ds[i][0][None].to(dev))
                    raw.append((pred[0].cpu(), logits[0].cpu()))
            return raw

        def gts_of(ds):
            return [[ds[i][1][j].numpy() for j in range(ds[i][2])] for i in range(len(ds))]

        raw_test, raw_val = raw_preds(test), raw_preds(val)
        gts_val = gts_of(val)

        def eval_at(raw, gts, thresh):
            dets_all = [learned_dets(p, l, thresh) for p, l in raw]
            n_det = sum(len(d) for d in dets_all)
            n_gt = sum(len(g) for g in gts)
            p, r, f = prf(dets_all, gts, args.family, args.size)
            return dict(thresh=thresh, n_det=n_det, n_gt=n_gt, P=p, R=r, F=f)

        grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
        val_sweep = [eval_at(raw_val, gts_val, t) for t in grid]
        test_sweep = [eval_at(raw_test, gts_by_img, t) for t in grid]
        if args.sweep:
            for name, sw in (("validation", val_sweep),
                             ("test (reference only, NOT used for selection)", test_sweep)):
                print(f"{name} sweep:")
                for r in sw:
                    print(f"  [{args.model}] thresh {r['thresh']:.2f}: det {r['n_det']} vs "
                          f"gt {r['n_gt']}  P {r['P']:.3f} R {r['R']:.3f} F {r['F']:.3f}")
        if args.thresh is not None:
            chosen, how = args.thresh, "fixed by --thresh"
        else:
            chosen, how = max(val_sweep, key=lambda r: r["F"])["thresh"], "selected on validation set"
        res = eval_at(raw_test, gts_by_img, chosen)
        oracle = max(test_sweep, key=lambda r: r["F"])
        print(f"[{args.model}] threshold {chosen:.2f} ({how})")
        print(f"[{args.model}] TEST (scored once): detections {res['n_det']} vs gt {res['n_gt']}  "
              f"P {res['P']:.3f} R {res['R']:.3f} F {res['F']:.3f}")
        print(f"[{args.model}] test-sweep oracle (old protocol, optimistic): best F "
              f"{oracle['F']:.3f} at thresh {oracle['thresh']:.2f}  "
              f"(gap vs validation-selected: {oracle['F'] - res['F']:+.3f})")
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(dict(args=vars(args), chosen_thresh=chosen, test=res,
                               test_sweep_oracle=oracle, val_sweep=val_sweep,
                               test_sweep=test_sweep), fh, indent=1, default=float)
        return

    n_det = sum(len(d) for d in dets_all)
    n_gt = sum(len(g) for g in gts_by_img)
    p, r, f = prf(dets_all, gts_by_img, args.family, args.size)
    print(f"[{args.model}] detections {n_det} vs gt {n_gt}  P {p:.3f} R {r:.3f} F {f:.3f}"
          "  (untuned: always keeps the top-4 candidates)")
    out = dict(args=vars(args), test=dict(n_det=n_det, n_gt=n_gt, P=p, R=r, F=f))
    if args.tune_score:
        val = SyntheticCurves(args.family, args.size, args.n_val, seed=3 + base, **kw)
        gts_val = [[val[i][1][j].numpy() for j in range(val[i][2])] for i in range(len(val))]
        val_dets = rule_dets(val)
        scores = np.array([d[-1] for per in val_dets for d in per])
        cands = np.unique(np.quantile(scores, np.linspace(0.0, 0.95, 20)))

        def at(dets, gts, thr):
            filt = [[d for d in per if d[-1] >= thr] for per in dets]
            pp, rr, ff = prf(filt, gts, args.family, args.size)
            return dict(thresh=float(thr), n_det=sum(len(x) for x in filt),
                        n_gt=sum(len(g) for g in gts), P=pp, R=rr, F=ff)

        val_sweep = [at(val_dets, gts_val, t) for t in cands]
        test_sweep = [at(dets_all, gts_by_img, t) for t in cands]
        chosen = max(val_sweep, key=lambda r_: r_["F"])["thresh"]
        res = at(dets_all, gts_by_img, chosen)
        oracle = max(test_sweep, key=lambda r_: r_["F"])
        print(f"[{args.model}] min score {chosen:.3g} (selected on validation set)")
        print(f"[{args.model}] TEST tuned (scored once): detections {res['n_det']} vs gt "
              f"{res['n_gt']}  P {res['P']:.3f} R {res['R']:.3f} F {res['F']:.3f}")
        print(f"[{args.model}] test-sweep oracle: best F {oracle['F']:.3f} "
              f"(gap vs validation-selected: {oracle['F'] - res['F']:+.3f})")
        out.update(tuned=res, chosen_score=chosen, test_sweep_oracle=oracle,
                   val_sweep=val_sweep, test_sweep=test_sweep)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
