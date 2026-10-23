"""Run the full TuSimple evaluation: learned model and tuned RANSAC, for the
line and lane families, over several seeds, then print a summary table.

Every run is a separate subprocess that writes <out>/<name>.json. A run whose
json already exists is skipped, so a crashed session can be restarted with the
same command.

Usage (Colab):
    python run_real_suite.py --root /content/tusimple/TUSimple/train_set \
        --out /content/drive/MyDrive/curve_hough_runs/real --seeds 0 1 2 --epochs 8
Summary only (no training):
    python run_real_suite.py --root x --out <dir> --seeds 0 1 2 --summary-only
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, out_json):
    if os.path.exists(out_json):
        print(f"skip (exists): {out_json}")
        return
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"!! failed ({r.returncode}); not recorded, rerun to retry")


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def summarize(out, seeds, families, lse):
    rows = []
    def agg(name, vals, extra=""):
        vals = [v for v in vals if v is not None]
        if vals:
            rows.append(f"| {name} | {np.mean(vals):.3f} +/- {np.std(vals):.3f} | {len(vals)} | {extra} |")
    for fam in families:
        learned = [load(f"{out}/{fam}_max_s{s}.json") for s in seeds]
        agg(f"{fam}: factorized DHT (max pool)" if fam == "lane" else f"{fam}: factorized DHT",
            [r["test"]["F"] if r else None for r in learned],
            "thresholds: " + ",".join(f"{r['chosen_thresh']:.2f}" for r in learned if r))
        if fam == "lane" and lse:
            agg("lane: factorized DHT (log-sumexp)",
                [(load(f"{out}/lane_lse_s{s}.json") or {}).get("test", {}).get("F") for s in seeds])
        rr = [load(f"{out}/ransac_{fam}_s{s}.json") for s in seeds]
        agg(f"{fam}: RANSAC, tuned", [r["tuned"]["F"] if r else None for r in rr],
            "min counts: " + ",".join(str(r["chosen_min_count"]) for r in rr if r))
        agg(f"{fam}: RANSAC, untuned top-4", [r["untuned"]["F"] if r else None for r in rr])
    print("\n| method | test F (mean +/- std) | seeds | notes |\n|---|---|---|---|")
    print("\n".join(rows))
    print("\nthreshold/min-count selected on validation frames, test frames scored once;"
          " same split for all methods within a seed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--families", nargs="+", default=["line", "lane"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--lane-lse", action="store_true",
                    help="also train the lane model with log-sumexp pooling")
    ap.add_argument("--no-ransac", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.summary_only:
        mi = ["--max-images", str(a.max_images)] if a.max_images else []
        for s in a.seeds:
            for fam in a.families:
                oj = f"{a.out}/{fam}_max_s{s}.json"
                run([sys.executable, "-u", "train_real.py", "--root", a.root, "--family", fam,
                     "--pool", "max", "--epochs", str(a.epochs), "--seed", str(s),
                     "--json-out", oj] + mi, oj)
                if fam == "lane" and a.lane_lse:
                    oj = f"{a.out}/lane_lse_s{s}.json"
                    run([sys.executable, "-u", "train_real.py", "--root", a.root, "--family", fam,
                         "--pool", "logsumexp", "--epochs", str(a.epochs), "--seed", str(s),
                         "--json-out", oj] + mi, oj)
                if not a.no_ransac:
                    oj = f"{a.out}/ransac_{fam}_s{s}.json"
                    run([sys.executable, "-u", "real_data/ransac_real_baseline.py", "--root", a.root,
                         "--family", fam, "--seed", str(s), "--json-out", oj] + mi, oj)
    summarize(a.out, a.seeds, a.families, a.lane_lse)


if __name__ == "__main__":
    main()
