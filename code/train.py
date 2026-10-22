"""Train the factorized DHT on synthetic curves and evaluate with curve-EA.

Evaluation protocol (v24): the detection-score threshold is selected on a
held-out VALIDATION set (seed 3), and the TEST set (seed 2) is scored once at
that threshold. Earlier versions swept thresholds directly on the test set and
reported the best F; that number is still printed, clearly labelled as
"test-sweep oracle", so the size of the optimism is visible.

Usage:
    python train.py --family parabola --epochs 5 --n-train 2000
    python train.py --family line --epochs 5 --n-train 2000
    python train.py --family parabola --size 64 --load parabola_hard.pt
    python train.py ... --thresh 0.35        # fixed threshold, no selection
    python train.py ... --json-out run.json  # machine-readable results
Smoke test:
    python train.py --smoke
    python train.py --family line --smoke
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticCurves
from hough import FactorizedDHT, stage1_loss
from metrics import prf

THRESH_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    params = torch.stack([b[1] for b in batch])
    counts = [b[2] for b in batch]
    return imgs, params, counts


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="parabola",
                    choices=["parabola", "circle", "ellipse", "lane", "line"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200,
                    help="validation images used ONLY to pick the threshold")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0,
                    help="model init / shuffling seed AND dataset seed offset "
                         "(train/test/val data seeds are 1/2/3 + 10*seed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--easy", action="store_true",
                    help="no noise/occlusion/distractors, single curve")
    ap.add_argument("--noise", type=float, default=0.08,
                    help="std of additive Gaussian pixel noise")
    ap.add_argument("--occ-max", type=int, default=2,
                    help="max number of occluding rectangles per image "
                         "(count drawn uniformly from 0..occ-max)")
    ap.add_argument("--pool", default="logsumexp",
                    choices=["logsumexp", "mean", "max"],
                    help="Stage-1 pooling over probe shapes")
    ap.add_argument("--probe-shapes", type=int, default=6,
                    help="Stage-1 probe set size (parabola/lane/circle)")
    ap.add_argument("--ellipse-probe-per-axis", type=int, default=3,
                    help="Stage-1 probes per axis for ellipse (27 at default 3)")
    ap.add_argument("--thresh", type=float, default=None,
                    help="fixed detection threshold; if omitted the threshold "
                         "is selected on the validation set")
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--resume", default="",
                    help="load weights and continue training for --epochs more "
                         "epochs (unlike --load, which skips straight to eval)")
    ap.add_argument("--sweep", action="store_true",
                    help="also print the full validation and test threshold sweeps")
    ap.add_argument("--topk", type=int, default=None,
                    help="max detections per image (defaults to model's, 4)")
    ap.add_argument("--diag", action="store_true",
                    help="print matched curve-EA similarity distribution")
    ap.add_argument("--json-out", default="",
                    help="write results (args, sweeps, chosen threshold, test "
                         "metrics) to this JSON file")
    return ap


def build_model(args, dev):
    is_line = (args.family == "line")
    model_kwargs = dict(topk=args.topk if args.topk else 4, pool=args.pool)
    if not is_line:
        model_kwargs["anchor_bins"] = 24 if args.smoke else 32
        model_kwargs["probe_shapes"] = args.probe_shapes
        model_kwargs["ellipse_probe_per_axis"] = args.ellipse_probe_per_axis
    else:
        model_kwargs["theta_bins"] = 24 if args.smoke else 60
        model_kwargs["r_bins"] = 24 if args.smoke else 60
    return FactorizedDHT(args.family, args.size, **model_kwargs).to(dev)


def all_detections(model, ds, dev):
    """Run the network once per image with thresh=0 (keeps every top-k peak
    with its score). Any threshold is then just a filter on the score, which
    is exactly equivalent to calling detect(thresh=t) but needs one forward
    pass per image instead of one per threshold."""
    dets, gts = [], []
    for i in range(len(ds)):
        img, params, k = ds[i]
        dets.append(model.detect(img[None].to(dev), thresh=0.0)[0])
        gts.append([params[j].numpy() for j in range(k)])
    return dets, gts


def score_at(dets, gts, family, size, thresh, return_sims=False):
    filt = [[d for d in per if d[-1] >= thresh] for per in dets]
    n_det = sum(len(d) for d in filt)
    n_gt = sum(len(g) for g in gts)
    (p, r, f), sims = prf(filt, gts, family, size, return_sims=True)
    out = dict(thresh=thresh, n_det=n_det, n_gt=n_gt, P=p, R=r, F=f)
    return (out, sims) if return_sims else out


def run(args):
    if args.smoke:
        args.epochs, args.n_train, args.n_test, args.n_val, args.size = 1, 16, 8, 8, 64

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    is_line = (args.family == "line")
    kw = dict(noise=0.0, occlude=False, distractors=False, max_curves=1) \
        if args.easy else dict(noise=args.noise, occ_max=args.occ_max)
    base = 10 * args.seed
    train = SyntheticCurves(args.family, args.size, args.n_train, seed=1 + base, **kw)
    val = SyntheticCurves(args.family, args.size, args.n_val, seed=3 + base, **kw)
    test = SyntheticCurves(args.family, args.size, args.n_test, seed=2 + base, **kw)
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    model = build_model(args, dev)

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
                    # a line has no Stage 2 (no spatial anchor to condition
                    # a shape accumulator on), so only Stage 1's loss applies
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
        if args.save:
            print(f"training complete, final weights in {args.save}")

    result = evaluate(model, args, val, test, dev)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=1, default=float)
        print(f"results written to {args.json_out}")
    return model, result, (val, test)


def evaluate(model, args, val, test, dev, verbose=True):
    """Select the threshold on `val`, score `test` once at that threshold, and
    also compute the old test-sweep oracle for comparison. Anchor recall
    (Stage-1 top-k, one-to-one, tol 1 bin) is attached for non-line families."""
    model.eval()
    val_dets, val_gts = all_detections(model, val, dev)
    test_dets, test_gts = all_detections(model, test, dev)
    val_sweep = [score_at(val_dets, val_gts, args.family, args.size, t) for t in THRESH_GRID]
    test_sweep = [score_at(test_dets, test_gts, args.family, args.size, t) for t in THRESH_GRID]

    if args.thresh is not None:
        chosen, how = args.thresh, "fixed by --thresh"
    else:
        best_val = max(val_sweep, key=lambda r: r["F"])   # first max wins ties
        chosen, how = best_val["thresh"], "selected on validation set"

    test_res, sims = score_at(test_dets, test_gts, args.family, args.size, chosen,
                              return_sims=True)
    oracle = max(test_sweep, key=lambda r: r["F"])

    if verbose:
        if args.sweep:
            print("validation sweep:")
            for r in val_sweep:
                print(f"  thresh {r['thresh']:.2f}: det {r['n_det']} vs gt {r['n_gt']}  "
                      f"P {r['P']:.3f} R {r['R']:.3f} F {r['F']:.3f}")
            print("test sweep (for reference only, NOT used for selection):")
            for r in test_sweep:
                print(f"  thresh {r['thresh']:.2f}: det {r['n_det']} vs gt {r['n_gt']}  "
                      f"P {r['P']:.3f} R {r['R']:.3f} F {r['F']:.3f}")
        print(f"threshold {chosen:.2f} ({how})")
        print(f"TEST (scored once): detections {test_res['n_det']} vs gt {test_res['n_gt']}  "
              f"P {test_res['P']:.3f} R {test_res['R']:.3f} F {test_res['F']:.3f}")
        print(f"test-sweep oracle (old protocol, optimistic): best F {oracle['F']:.3f} "
              f"at thresh {oracle['thresh']:.2f}  "
              f"(gap vs validation-selected: {oracle['F'] - test_res['F']:+.3f})")
        if args.diag and len(sims):
            q = np.percentile(sims, [10, 25, 50, 75, 90])
            print(f"  matched curve-EA similarity: mean {sims.mean():.3f}  "
                  f"p10/p25/p50/p75/p90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}/"
                  f"{q[3]:.2f}/{q[4]:.2f}  (n={len(sims)})")

    result = dict(args=vars(args), chosen_thresh=chosen, threshold_source=how,
                  topk=model.topk, test=test_res, test_sweep_oracle=oracle,
                  val_sweep=val_sweep, test_sweep=test_sweep)
    if args.family != "line":
        import anchor_eval as A          # lazy: anchor_eval imports this module
        rec, _ = A.topk_anchor_recall(model, test, dev, ks=(1, 2, 4, 8), tols=(1,))
        result["anchor_recall_tol1"] = {str(k): v for k, v in rec[1].items()}
        if verbose:
            print("Stage-1 top-k anchor recall (tol 1 bin, one-to-one): " +
                  "  ".join(f"k={k}: {v:.3f}" for k, v in rec[1].items()))
    return result


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
