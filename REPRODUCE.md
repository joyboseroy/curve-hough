# Reproducing every number in the paper

Every figure in every table comes from one of the commands below, run from
`code/`. Seeds are fixed in the scripts (torch/numpy seed 0; dataset seeds
1/2 for train/test), so reruns reproduce the reported numbers up to
hardware nondeterminism. Thresholds in the paper are the best-F entries of
each printed sweep.

## Evaluation protocol v24 (supersedes the threshold rule above)
The numbers in the first submission picked each detection threshold by
sweeping on the test set and reporting the best F. That is test-set tuning
and makes the F-scores optimistic. From v24 on:
  * the threshold is selected on a held-out validation set (dataset seed
    3 + 10*seed), and the test set (seed 2 + 10*seed) is scored once at that
    threshold, for our model (train.py) and for the learned baselines
    (eval_baselines.py). Rule-based baselines have no score threshold, so
    they are unchanged.
  * every run also prints "test-sweep oracle", the old best-of-sweep number,
    and its gap to the validation-selected number, so the optimism of the old
    protocol is measured, not assumed.
  * all tables must be regenerated; the old values are not comparable.
  * use several seeds (--seed 0 1 2 ...); with 200 test images, single-run
    differences of a few F points are within noise.

    # one run, machine-readable (paper setting)
    python train.py --family parabola --size 64 --epochs 10 --n-train 1000 \
        --n-val 200 --n-test 200 --save parabola_hard.pt --json-out parabola_hard.json
    # same protocol for the learned baselines
    python eval_baselines.py --model regression --family parabola --epochs 10 \
        --n-train 1000 --n-val 200 --n-test 200 --json-out reg.json
    python eval_baselines.py --model query --family parabola --epochs 10 \
        --n-train 1000 --n-val 200 --n-test 200 --json-out query.json

    # rule-based baselines get the same treatment: a minimum detection score
    # (RANSAC inlier count, Hough accumulator value) selected on validation.
    # Without --tune-score they always keep their top-4 candidates, which caps
    # precision and is not comparable with a thresholded model.
    python eval_baselines.py --model ransac --family parabola --n-val 200 \
        --n-test 200 --tune-score --json-out ransac_tuned.json
    python eval_baselines.py --model classical --family parabola --n-val 200 \
        --n-test 200 --tune-score --json-out classical_tuned.json
    python eval_baselines.py --model ransac --family ellipse --n-val 200 \
        --n-test 200 --tune-score --json-out ransac_ellipse_tuned.json

### Ablations (Reviewer 1 and 2 requests)
    python ablate.py --suite pool probes topk noise occlusion \
        --family parabola --size 64 --epochs 10 --n-train 1000 \
        --n-val 200 --n-test 200 --seeds 0 1 2 --out ablations_parabola
Writes ablations_parabola.json and .md (mean and std over seeds). Results are
saved after every finished run; rerun with --resume to continue after a crash
and --values to restrict the grid (e.g. --suite noise --values 0.0 0.16 0.24).
Choose a configuration by validation F (the val F column), never by test F. The topk
suite re-evaluates one trained model per seed at k = 1, 2, 4, 8, so it costs
no extra training. Cost: 17 trained models per seed for the other suites.

### Stage-1 anchor recall and anchor separation
    python anchor_eval.py --family parabola --size 64 --load parabola_hard.pt \
        --thresh-from parabola_hard.json --json-out anchors_parabola.json
Top-k anchor recall (one-to-one, tolerance in anchor bins) measures how
precisely Stage 1 localizes anchors; it is not a bound on end-to-end recall,
because F/R average over lenient curve-similarity thresholds. The script also
prints an oracle-anchor ceiling: Stage 2 run at the ground-truth anchors,
compared with the real pipeline. That difference is the cost of Stage 1. The separation sweep uses two curves
whose anchors are 0 to 24 px apart; use a model trained on the multi-curve
(hard) setting, since a single-curve model is out of distribution there.

## Table 1 (accumulator memory, d=3) and Table 2 (memory scaling d=3 vs d=5)
    python measure_memory.py
No training. Prints both tables' numbers, including the arithmetic
(not instantiated) dense d=5 size.

## Table 3 (parabola, corrupted + clean)
    # corrupted ("main") row + threshold sweep; paper reports thresh 0.55
    python train.py --family parabola --size 64 --epochs 10 --n-train 1000 \
        --n-test 200 --save parabola_hard.pt --sweep
    # matched-similarity diagnostics (median / p10 columns)
    python train.py --family parabola --size 64 --load parabola_hard.pt \
        --thresh 0.55 --diag
    # clean-control row (evaluates the hard-trained model on clean data)
    python train.py --family parabola --size 64 --load parabola_hard.pt \
        --easy --thresh 0.55 --diag

## Table 4 (circle, corrupted)
    python train.py --family circle --size 64 --epochs 10 --n-train 1000 \
        --n-test 200 --save circle_hard.pt --sweep
Paper reports the best-F sweep entry (thresh 0.75).

## Table 5 (baselines, corrupted parabola)
    python eval_baselines.py --model classical --family parabola --n-test 200
    python eval_baselines.py --model ransac --family parabola --n-test 200
    python eval_baselines.py --model regression --family parabola \
        --epochs 10 --n-train 1000 --n-test 200 --sweep
    python eval_baselines.py --model query --family parabola \
        --epochs 10 --n-train 1000 --n-test 200 --sweep
Paper reports classical/RANSAC as printed, and the best-F sweep entries for
regression (thresh 0.80) and query (thresh 0.60).

## Table 6 (ellipse, clean + corrupted)
    # clean row (thresh 0.55 best of sweep)
    python train.py --family ellipse --size 64 --epochs 10 --n-train 1000 \
        --n-test 200 --easy --save ellipse_easy.pt --sweep
    # corrupted row (thresh 0.35 best of sweep)
    python train.py --family ellipse --size 64 --epochs 10 --n-train 1000 \
        --n-test 200 --save ellipse_hard.pt --sweep

## Figures
    # Figure 1 (accumulator schematic; conceptual, matched to Table 1 numbers)
    python make_schematic_figure.py
    # Figure 2 (benchmark examples; pure generator output)
    python make_benchmark_examples.py
    # Figures 3-4 (qualitative + accumulator/profile views, from real checkpoints)
    python make_figures.py --family parabola --load parabola_hard.pt --idx 3
    python make_figures.py --family ellipse --load ellipse_hard.pt --idx 0

## Practical notes
- CPU runs: ~1-2 h per training command on a laptop; --save checkpoints
  every epoch, and --resume continues after a disconnect (see README).
- The ellipse commands are the slowest (larger probe/dense shape grids).
- Numbers not yet in the paper (and so not reproducible from it): circle
  clean control, circle/ellipse baselines, ablations, 128x128 runs. These
  are listed as remaining work in the README, not reported anywhere.

## TuSimple under the v24 protocol

Frames are split 80/10/10 train/val/test with a seeded shuffle (`code/real_data/split.py`).
The threshold (model) or minimum inlier count (RANSAC) is selected on the validation
frames and the test frames are scored once. Lane models use max pooling.

    cd code
    python run_real_suite.py --root /path/to/TUSimple/train_set \
        --out real_runs --seeds 0 1 2 --epochs 8 --lane-lse

writes one json per run and prints a summary table (mean +/- std over seeds) for the
learned line and lane models and for tuned and untuned RANSAC. Single runs:
`train_real.py --seed S --json-out f.json` and `real_data/ransac_real_baseline.py --seed S --json-out f.json`.
`real_data/ransac_line_baseline.py` is the older untuned, test-only version.
