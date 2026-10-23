# Results ledger (v24 protocol)

Setting for every row: synthetic corrupted benchmark (noise 0.08, up to 3 curves,
distractors, occluders), 64 px, 1000 train images, 10 epochs, 200 validation and
200 test images. Detection thresholds are selected on the validation set and the
test set is scored once. Unless stated, seed 0 (the same 200 test images for
every method). F is the curve-EA based F used throughout the paper. Differences
below about 0.02 are within test-set noise at this size.

## Headline comparison (seed 0)

| Family | Factorized DHT, max pool | Factorized DHT, logsumexp | RANSAC tuned | RANSAC untuned | Classical Hough tuned |
|---|---|---|---|---|---|
| Parabola (d=3) | 0.706 | 0.519 | 0.702 | 0.551 | 0.609 (16 anchor bins) |
| Circle (d=3) | 0.743 | not rerun | 0.742 | 0.584 | not run |
| Ellipse (d=5) | 0.382 | 0.319 | 0.410 | 0.383 | not run (dense 5-D infeasible) |

Learned free-form baselines (parabola): regression 0.151, DETR-lite queries 0.131.
Not yet run: regression and DETR-lite for circle and ellipse.

"Tuned" means a minimum detection score (RANSAC inlier count, Hough accumulator
value) selected on validation, the same treatment the learned models get.
"Untuned" always keeps the top 4 candidates. Classical Hough is run at 16 anchor
bins and 16 shape bins (the cascade uses 32 anchor bins), so it is handicapped.

## Pooling (parabola, 3 seeds, mean +/- std)

| Stage-1 pooling | val F | test F | anchor recall@4 (1 bin) |
|---|---|---|---|
| max | 0.697 +/- 0.024 | 0.696 +/- 0.011 | 0.765 |
| logsumexp | 0.533 +/- 0.016 | 0.511 +/- 0.006 | 0.494 |

Max was chosen by validation F on seeds 1 and 2 (0.710 vs 0.539), then seed 0
was added. File: abl_pool_seeds.json.

## Stage-1 diagnostics (parabola, max pool, seed 0; parabola_max.json, anchors_parabola_max.json)

Anchor recall (one-to-one), tolerance in anchor bins (1 bin is about 2 px):

| tol | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| 0 | 0.189 | 0.263 | 0.288 | 0.291 |
| 1 | 0.452 | 0.684 | 0.796 | 0.811 |
| 2 | 0.472 | 0.724 | 0.883 | 0.903 |

Oracle-anchor ceiling (Stage 2 at ground-truth anchors): F 0.938, against 0.706
for the real pipeline. The cost of Stage 1 is 0.232 F. Stage 2 is not the limit.

Anchor separation (two curves, noise only, threshold 0.55):

| sep (px) | 0 | 2 | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|
| anchor recall@4 | 0.475 | 0.482 | 0.492 | 0.525 | 0.568 | 0.812 | 0.870 | 0.912 |
| F | 0.584 | 0.588 | 0.584 | 0.582 | 0.640 | 0.746 | 0.779 | 0.809 |

Anchors closer than about 8 px (about 4 anchor bins) cost roughly 0.2 F, the
point where the 3x3 peak window merges them.

## Protocol effect
Validation-selected vs old best-of-test-sweep F differs by 0.000 to 0.011 across
all runs; the old protocol was procedurally wrong but moved numbers little.

## Still to do
Ablations under max pooling (topk, noise, occlusion, probes), baselines for
circle and ellipse, multi-seed circle and ellipse, real dense d=4 comparison,
multi-seed TuSimple with published baselines, a fix for close anchors.

## TuSimple (v24 protocol, 3 seeds, 8 epochs, 128 px, random frame split 80/10/10)

| method | test F (mean +/- std) | per seed |
|---|---|---|
| line: factorized DHT | 0.614 +/- 0.019 | 0.633, 0.589, 0.622 |
| line: RANSAC tuned / untuned | 0.252 +/- 0.001 | 0.253, 0.251, 0.253 |
| lane: factorized DHT (max) | 0.201 +/- 0.004 | 0.198, 0.207, 0.199 |
| lane: factorized DHT (log-sumexp) | 0.177 +/- 0.012 | 0.187, 0.185, 0.160 |
| lane: RANSAC tuned | 0.136 +/- 0.008 | 0.135, 0.127, 0.147 |
| lane: RANSAC untuned | 0.134 +/- 0.006 | 0.138, 0.126, 0.140 |

Lane anchor recall@4 (1 bin): max 0.147, log-sumexp 0.107. Lane threshold selected at the grid floor (0.05) in 2 of 3 seeds for max pooling.
Caveats: split by frame not clip, generic Sobel RANSAC without road mask, two separate models, no published detectors.

## Memory, matched grid
Cascade ellipse shape set is 6^3 = 216 hypotheses (parabola, circle: 48), not 32 bins per shape axis. Dense at B=32 on every axis: 4 GB vs 236 KB (17,772x). Dense on the cascade's own grid (32x32 anchors x 216 shapes x C=32, float32): 27.0 MB, a like-for-like ratio of 117x (parabola: 6.00 MB vs 152 KB, 40.4x).
