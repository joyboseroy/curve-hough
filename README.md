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

## Change history (chronological)

### v12: ellipse support + a real memory bug found and fixed
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

### v14: checkpoint-every-epoch + resume, and a paper-vs-code honesty fix
A long run timing out used to lose all progress (train.py only saved at the
very end). Now `--save` checkpoints after every epoch, and `--resume
<path>` loads a checkpoint and continues training for `--epochs` more
epochs (unlike `--load`, which skips straight to eval). If a run times out,
rerun with `--resume` on the same save path instead of starting over.

Also: a review correctly caught that Section 3.4 and the Limitations
paragraph described an "optional refinement stage" (re-voting on a newly
built finer grid, the coarse-to-fine classical Hough analogue) as if it
were implemented and ablatable. It isn't -- only cheap soft-argmax
interpolation over the existing coarse grid is implemented (and that part
of the review's other claims -- logsumexp pooling, NMS, soft-argmax decode
-- were verified as genuinely implemented, so the review was right about
one specific thing and overstated on the rest). Fixed: the paper now
clearly separates "soft-argmax interpolation (implemented)" from "coarse-
to-fine re-voting (not implemented, planned)", and the impossible ablation
(vi) is removed rather than left in the results plan.

### v16: figures
Two kinds, handled differently -- an image generator (DALL-E-style) is the
wrong tool for either; both are built as real matplotlib/vector graphics.

- `code/make_schematic_figure.py`: conceptual dense-vs-factorized
  accumulator diagram, no data dependency, numerically matched to the
  measured Table 1 numbers (26.9x, 152KB vs 4MB). Already generated and
  inserted into the paper (Figure 1, Section 3.4).
- `code/make_figures.py`: data-driven figures that MUST come from a real
  trained checkpoint -- Stage-1 accumulator heatmap (with GT and detected
  peaks marked), Stage-2 shape profile at the top peak, and a qualitative
  image with ground truth vs. detected curves overlaid. Validated the
  script runs cleanly (parabola and ellipse, including the multi-dim shape
  path) against throwaway smoke checkpoints, but that output was NOT used
  in the paper -- it's an untrained toy model, not a real result, same
  discipline as everywhere else in this project. Run it yourself against a
  real checkpoint:
  `python make_figures.py --family parabola --load parabola_hard.pt --idx 3`
  (pick --idx to find a representative example; --thresh may need lowering
  if no detections show up at the default 0.25). Copy the three PNGs it
  produces into paper/figures/, then uncomment the placeholder figure block
  in main.tex (search "Qualitative results") and add one more per family.

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
   evidence the factorization scales for clean-setting accuracy.
   DONE: ellipse hard-setting run. Best F 0.329 @ thresh=0.35 (P 0.350,
   R 0.311) -- clean-to-corrupted drop is 0.352 (51.7% relative), nearly
   3x parabola's drop (0.125, 19.2% relative). Real, honest finding: ellipse
   is tractable and accurate when clean but degrades much faster under
   occlusion/clutter than lower-d families -- makes sense (occluding an arc
   constrains a joint (rx,ry,phi) far more weakly than a single radius).
   Written up in the paper as a genuine limitation, not smoothed over.
   Both runs now in the paper (Table: ellipse_interim) with the
   clean-vs-corrupted comparison spelled out.
3. Ellipse baselines: RANSAC has a closed-form 5-point conic fit (real,
   worth implementing properly, not a stub); classical dense Hough at d=5
   is likely impractical at any reasonable resolution -- probably worth
   running once at a small resolution just to show it's impractical
   (or timing out), which is itself evidence for the paper's motivation.
   Direct regression / DETR-lite queries need their output width widened
   from 4 (x0,y0,shape,presence) to 6 (x0,y0,rx,ry,phi,presence).
4. Ablations: probe set size m, top-k, max vs mean vs logsumexp pooling.
   (Coarse-to-fine re-voting refinement is not implemented -- see item 9 --
   so it's future work, not an ablation on the current codebase.)
5. Occlusion/clutter parametric sweep (0/20/40/60%), not just easy-vs-hard.
6. Verify two flagged citations in main.tex (algebraic-curve HT + CT
   application; HoughLaneNet and BSNet author lists).
7. Qualitative figure + benchmark examples figure; a schematic diagram of
   dense vs. factorized accumulator (reviewer-requested, conceptual only).
8. 128x128 scale-up on Colab for final paper numbers (64px numbers so far
   are real but a smaller-scale stand-in).
9. Coarse-to-fine re-voting refinement (Section 3.4): build a finer local
   grid around each detection and re-run the vote/gather at that
   resolution, rather than the soft-argmax interpolation currently
   implemented. Real implementation work, not yet started; the paper
   currently describes this honestly as planned, not shipped.

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
