# Reproducing every number in the paper

Every figure in every table comes from one of the commands below, run from
`code/`. Seeds are fixed in the scripts (torch/numpy seed 0; dataset seeds
1/2 for train/test), so reruns reproduce the reported numbers up to
hardware nondeterminism. Thresholds in the paper are the best-F entries of
each printed sweep.

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
