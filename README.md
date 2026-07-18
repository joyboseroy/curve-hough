# Beyond Lines: Factorized Deep Hough Transform for Parametric Curves

Fast-track arXiv paper project. Generalizes the Deep Hough Transform (Zhao et
al., TPAMI 2021) from straight lines to parametric curve families via a
cascade of low-dimensional, feature-space voting stages.

## Layout
- paper/main.tex : full paper skeleton (abstract, method, propositions,
  attention-comparison table, pseudocode, all done; results tables have real
  measured numbers for parabola/circle + baselines; ellipse results TODO)
- code/dataset.py : synthetic benchmark (parabola, circle, ellipse; noise,
  occlusion, distractor segments). Params tensor always width 5
  (x0, y0, shape...), padded with zeros for lower-dimensional families.
- code/hough.py : curve banks (family-generic, shape is always a tuple),
  parameter-free voting layer, chunked stage1_accumulate (bounds peak
  memory -- see v12 note below), FactorizedDHT (stage 1 marginal anchor
  voting + stage 2 conditional shape voting, both shape-dim-agnostic),
  bounded LRU cache for stage-2 banks, stage-1/stage-2 losses
- code/metrics.py : curve-EA score (parabola/circle/ellipse) and
  Hungarian-matched P/R/F evaluation
- code/baselines.py : classical dense Hough on edges, RANSAC, direct
  regression head, DETR-lite hypothesis-query head -- parabola/circle only;
  ellipse baselines not yet implemented (see Remaining work)
- code/train.py : training and evaluation entry point (--family parabola |
  circle | ellipse); --save/--load/--sweep/--diag/--topk
- code/eval_baselines.py : baseline training/eval harness (parabola/circle
  only); --save/--load/--sweep
- code/measure_memory.py : real accumulator memory measurement, no training

## Status
Smoke tests pass end to end for all three families:
`python train.py --smoke`, `--family circle --smoke`, `--family ellipse --smoke`

## v12: ellipse support + a real memory bug found and fixed
Ellipse is d=5 (x0, y0, rx, ry, phi); shape is now a tuple everywhere
(dataset, curve_pixels, build_bank, FactorizedDHT, metrics.curve_ea) instead
of the scalar it was for parabola/circle -- parabola/circle behavior is
unchanged (verified: smoke-test losses are bit-identical to the pre-ellipse
version).

While bringing up ellipse, `vote()` was found to materialize one gather
tensor across ALL probe shapes simultaneously: size scales with
B * C * A * S * max_pts. This was always somewhat wasteful but became a real
problem with ellipse's larger probe count (27 vs 6 for parabola) -- it was
hitting ~2GB for a single forward call and getting OOM-killed. Fixed with
`stage1_accumulate()`, which loops over probe shapes one at a time so peak
memory no longer scales with S. This is a general fix, not ellipse-specific,
and should also help parabola/circle at larger scale (128px, bigger batches).

Also added a bounded LRU cache for stage-2 banks (`bank2_cache_size`,
default 64) so memory doesn't grow unbounded as training visits more
distinct anchor bins over an epoch.

## Validation protocol (run in this order, keep all real numbers)
1. `python train.py --family parabola --size 64 --epochs 10 --n-train 1000
   --n-test 200 --easy`  -> algorithm sanity: expect high F (0.8+). If not,
   debug before anything else.
2. Same without --easy -> the honest hard-setting number.
3. Then circles, then baselines, then 128 px on Colab for the paper table.
4. Ellipse: same protocol, family=ellipse. Not yet run at full scale --
   next thing to do. Ellipse construction/training is slower per-step than
   parabola/circle (bigger probe/dense grids), budget more wall-clock time.

IMPORTANT: results tables in the paper must contain only numbers produced by
these runs. Do not paste numbers from any external draft or suggestion.

## Remaining work (ordered, working through one at a time)
1. DONE: ellipse family support (dataset, model, metric) + memory fix.
2. DONE: ellipse easy/clean training run. Best F 0.681 @ thresh=0.55
   (P 0.690, R 0.672), losses converged cleanly, detection counts track GT
   closely (no over/under-detection). Notably beats parabola's own easy
   number (F 0.650) despite ellipse's larger 5-parameter space -- good
   evidence the factorization scales.
   NEXT: ellipse hard-setting run (same protocol, drop --easy):
   `python train.py --family ellipse --size 64 --epochs 10 --n-train 1000
   --n-test 200 --save ellipse_hard.pt --sweep`
3. Ellipse baselines: RANSAC has a closed-form 5-point conic fit (real,
   worth implementing properly, not a stub); classical dense Hough at d=5
   is likely impractical at any reasonable resolution -- probably worth
   running once at a small resolution just to show it's impractical
   (or timing out), which is itself evidence for the paper's motivation.
   Direct regression / DETR-lite queries need their output width widened
   from 4 (x0,y0,shape,presence) to 6 (x0,y0,rx,ry,phi,presence).
4. Ablations: probe set size m, top-k, max vs mean vs logsumexp pooling,
   refinement stage on/off.
5. Occlusion/clutter parametric sweep (0/20/40/60%), not just easy-vs-hard.
6. Verify two flagged citations in main.tex (algebraic-curve HT + CT
   application; HoughLaneNet and BSNet author lists).
7. Qualitative figure + benchmark examples figure; a schematic diagram of
   dense vs. factorized accumulator (reviewer-requested, conceptual only).
8. 128x128 scale-up on Colab for final paper numbers (64px numbers so far
   are real but a smaller-scale stand-in).

## Design decisions already made
- Method core is geometric factorized voting, not learned queries. The
  transformer angle appears as (a) the discussion section: Hough voting =
  cross-attention with a geometry-fixed attention map, plus a comparison
  table, and (b) the DETR-lite baseline. Learned hypothesis spaces are
  future work, deliberately.
- Metric mirrors the DHT paper's EA score (contribution symmetry with the
  paper being generalized), generalized to arbitrary shape dimensionality.
- Paper's empirical framing is "geometric structure vs. free-form
  prediction," not "we beat RANSAC" -- supported by real baseline numbers,
  see main.tex Table 3 and its discussion.
