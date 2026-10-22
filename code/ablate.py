"""Ablation runner for the factorized DHT.

Covers the ablations the CODS 2026 reviewers asked for:
    pool       Stage-1 pooling over probe shapes (logsumexp | mean | max)
    probes     Stage-1 probe set size
    topk       number of retained Stage-1 peaks k (one trained model per seed,
               re-evaluated at each k, so it costs no extra training)
    noise      additive pixel-noise std (trained and tested at that level)
    occlusion  max number of occluding rectangles (trained and tested at that level)

Every cell uses the validation-selected threshold and a single test pass, and
reports mean +/- std over --seeds. Seeds change model init, shuffling AND the
train/val/test image draws (see train.py), so the std includes data variance.

Usage (parabola, paper setting, 3 seeds):
    python ablate.py --suite pool probes topk noise occlusion \
        --family parabola --size 64 --epochs 10 --n-train 1000 --seeds 0 1 2 \
        --out ablations_parabola
Cheap check that the plumbing works:
    python ablate.py --suite pool topk --smoke --seeds 0

Cost: each trained model is one full train.py run. pool(3) + probes(5) +
noise(5) + occlusion(4) = 17 trained models per seed; topk reuses a trained
model. Plan GPU time accordingly, or start with --seeds 0.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import torch

import train as T

PROBE_VALUES = {"parabola": [4, 6, 8, 12, 16], "lane": [4, 6, 8, 12, 16],
                "circle": [4, 6, 8, 12, 16], "ellipse": None}
ELLIPSE_PROBES = [2, 3, 4]          # per axis -> 8, 27, 64 probe shapes (2 per axis already spans both extremes)
TOPK_VALUES = [1, 2, 4, 8]
NOISE_VALUES = [0.0, 0.04, 0.08, 0.16, 0.24]
OCC_VALUES = [0, 1, 2, 4]
POOL_VALUES = ["logsumexp", "mean", "max"]


def suite_spec(name, family):
    if name == "pool":
        return "pool", POOL_VALUES
    if name == "probes":
        if family == "ellipse":
            return "ellipse_probe_per_axis", ELLIPSE_PROBES
        return "probe_shapes", PROBE_VALUES[family]
    if name == "noise":
        return "noise", NOISE_VALUES
    if name == "occlusion":
        return "occ_max", OCC_VALUES
    if name == "topk":
        return "topk", TOPK_VALUES
    raise ValueError(name)


def metrics_row(result, seed):
    t = result["test"]
    rec = result.get("anchor_recall_tol1", {})
    return dict(seed=seed, F=t["F"], P=t["P"], R=t["R"], thresh=result["chosen_thresh"],
                oracle_gap=result["test_sweep_oracle"]["F"] - t["F"],
                anchor_recall_at_k=rec.get(str(result["topk"]), float("nan")),
                val_F=max(r["F"] for r in result["val_sweep"]))


CASTERS = {"pool": str, "probe_shapes": int, "ellipse_probe_per_axis": int,
           "noise": float, "occ_max": int, "topk": int}


def load_state(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(prefix, state):
    """Written after EVERY finished run, so a crashed session loses at most the
    run in progress. Reruns with --resume skip anything already recorded."""
    with open(prefix + ".json", "w") as fh:
        json.dump(state, fh, indent=1)


def run_one(argv, parser, isolate):
    """Train and evaluate one configuration. With isolate=True (default) it runs
    train.py in a fresh subprocess, so memory is returned to the OS after every
    run and a crashed or killed run cannot take the others down with it.
    Returns the result dict, or None if the subprocess failed."""
    if not isolate:
        _, result, _ = T.run(parser.parse_args(argv))
        return result
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        subprocess.run([sys.executable, "-u", "train.py"] + argv + ["--json-out", tmp],
                       check=True, cwd=here)
        with open(tmp) as fh:
            return json.load(fh)
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"!! run failed ({exc}); not recorded, rerun with --resume to retry")
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_param_suite(name, flag, values, base_argv, seeds, parser, state, prefix,
                    isolate=True):
    entry = state.setdefault(name, dict(flag=flag, raw={}))
    cells = entry["raw"]
    for v in values:
        for s in seeds:
            if any(r["seed"] == s for r in cells.get(str(v), [])):
                print(f"skip {name}: {flag}={v} seed={s} (already recorded)")
                continue
            argv = base_argv + [f"--{flag.replace('_', '-')}", str(v), "--seed", str(s)]
            parser.parse_args(argv)          # fail fast on a bad argument
            print(f"\n=== {name}: {flag}={v} seed={s} ===")
            result = run_one(argv, parser, isolate)
            if result is None:
                continue
            cells.setdefault(str(v), []).append(metrics_row(result, s))
            save_state(prefix, state)
    return cells


def run_topk_suite(values, base_argv, seeds, parser, state, prefix):
    entry = state.setdefault("topk", dict(flag="topk", raw={}))
    cells = entry["raw"]
    for s in seeds:
        if all(any(r["seed"] == s for r in cells.get(str(k), [])) for k in values):
            print(f"skip topk: seed={s} (already recorded)")
            continue
        args = parser.parse_args(base_argv + ["--seed", str(s)])
        print(f"\n=== topk: training one model, seed={s} ===")
        model, _, (val, test) = T.run(args)
        dev = next(model.parameters()).device
        for k in values:
            model.topk = k
            print(f"--- evaluating with k={k}")
            result = T.evaluate(model, args, val, test, dev, verbose=False)
            cells.setdefault(str(k), []).append(metrics_row(result, s))
        save_state(prefix, state)
    return cells


def aggregate(cells):
    table = {}
    for v, rows in cells.items():
        table[v] = {}
        for key in rows[0]:
            if key == "seed":
                continue
            vals = np.array([r[key] for r in rows], dtype=float)
            table[v][key] = dict(mean=float(np.nanmean(vals)),
                                 std=float(np.nanstd(vals)), n=len(vals))
    return table


def to_markdown(name, flag, table):
    lines = [f"### Ablation: {name} ({flag})", "",
             "| value | test F | P | R | anchor recall@k | val F | selected thresh | oracle gap |",
             "|---|---|---|---|---|---|---|---|"]
    def f(m): return f"{m['mean']:.3f} ± {m['std']:.3f}"
    for v, m in table.items():
        lines.append(f"| {v} | {f(m['F'])} | {f(m['P'])} | {f(m['R'])} | "
                     f"{f(m['anchor_recall_at_k'])} | {f(m['val_F'])} | "
                     f"{m['thresh']['mean']:.2f} | {m['oracle_gap']['mean']:+.3f} |")
    n = next(iter(table.values()))["F"]["n"]
    lines += ["", f"mean ± std over {n} seed(s); thresholds chosen on validation, "
                  "test scored once. 'oracle gap' is how much the old "
                  "test-sweep protocol would have overstated F. If you use an "
                  "ablation to choose a configuration, choose by val F, not test F.", ""]
    return "\n".join(lines)


def main():
    parser = T.build_parser()
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--suite", nargs="+", required=True,
                    choices=["pool", "probes", "topk", "noise", "occlusion"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--values", nargs="+", default=None,
                    help="override the value grid for ONE suite, e.g. "
                         "--suite noise --values 0.0 0.04 0.16 0.24")
    ap.add_argument("--in-process", action="store_true",
                    help="run all trainings in this one process (default: one fresh "
                         "subprocess per run, which survives a crashed run)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse <out>.json: skip runs already recorded there")
    ap.add_argument("--out", default="ablations",
                    help="output prefix: writes <out>.json and <out>.md")
    known, rest = ap.parse_known_args()
    base_argv = [a for a in rest]
    probe = parser.parse_args(base_argv)
    if probe.family == "line":
        raise SystemExit("ablations are defined for the anchor families, not line")
    if probe.load:
        raise SystemExit("ablate.py trains its own models; do not pass --load")
    if known.values and len(known.suite) != 1:
        raise SystemExit("--values applies to a single --suite")

    state = load_state(known.out + ".json") if known.resume else {}
    for name in known.suite:
        flag, values = suite_spec(name, probe.family)
        if known.values:
            values = [CASTERS[flag](v) for v in known.values]
        if name == "topk":
            run_topk_suite(values, base_argv, known.seeds, parser, state, known.out)
        else:
            run_param_suite(name, flag, values, base_argv, known.seeds, parser,
                            state, known.out, isolate=not known.in_process)
        save_state(known.out, state)

    md = []
    for name, entry in state.items():
        md.append(to_markdown(name, entry["flag"], aggregate(entry["raw"])))
    with open(known.out + ".md", "w") as fh:
        fh.write("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"wrote {known.out}.json and {known.out}.md")


if __name__ == "__main__":
    main()
