# Beyond Lines: Factorized Deep Hough Transform for Parametric Curves

Generalizes the Deep Hough Transform (Zhao et
al., TPAMI 2021) from straight lines to parametric curve families via a
cascade of low-dimensional, feature-space voting stages.

## Layout
- paper/main.tex : full paper (abstract, method, propositions,
  attention-comparison table, pseudocode, synthetic results for all four
  families incl. line, real-data validation section). Two versions
  maintained: an anonymized two-column version and a non-anonymous
  single-column version (full un-trimmed prose, no page limit) for
  arXiv/Zenodo/personal site.
- code/dataset.py : synthetic benchmark (parabola, circle, ellipse, line;
  noise, occlusion, distractor segments). Params tensor always width 5
  (x0, y0, shape...), padded with zeros for lower-dimensional families.
- code/hough.py : curve banks (family-generic, shape is always a tuple),
  parameter-free voting layer, chunked stage1_accumulate (bounds peak
  memory -- see v12 note below), FactorizedDHT (stage 1 marginal anchor
  voting + stage 2 conditional shape voting, both shape-dim-agnostic),
  bounded LRU cache for stage-2 banks, stage-1/stage-2 losses. `line` is
  the degenerate case (theta, r direct voting, no Stage 2) -- see v22.
  All families store params uniformly as (x0, y0, shape...); no per-family
  coordinate special-casing anywhere (see v23's bug writeup for why this
  matters).
- code/metrics.py : curve-EA score (parabola/circle/ellipse/line) and
  Hungarian-matched P/R/F evaluation
- code/baselines.py : classical dense Hough on edges, RANSAC, direct
  regression head, DETR-lite hypothesis-query head -- parabola/circle only;
  ellipse baselines not yet implemented (see Remaining work)
- code/train.py : synthetic training and evaluation entry point
  (--family parabola | circle | ellipse | line); --save/--load/--sweep/
  --diag/--topk
- code/eval_baselines.py : baseline training/eval harness (parabola/circle
  only); --save/--load/--sweep
- code/measure_memory.py : real accumulator memory measurement, no training
- code/train_real.py : real-data training/eval entry point (--family
  lane | line), for TuSimple photographs letterboxed to 128px. Computes
  the shape_range from the actual training data rather than inheriting
  synthetic-tuned defaults; --vertex-margin controls the off-frame
  vertex filter (see v23); --diag prints the matched-similarity
  distribution.
- code/real_data/tusimple_loader.py : parses TuSimple's label JSON and
  fits each real lane to a curve. Splits real lanes into two families
  by fit stability: 'lane' (stable parabola fit) vs 'line' (near-
  straight fallback when the parabola's vertex-form conversion is
  unstable).
- code/real_data/tusimple_dataset.py : real-image Dataset wrapping the
  loader -- letterbox transform (uniform scale + pad, preserves line
  angles and keeps the parameter transform closed-form), and the
  vertex-visibility filter (see v23).
- code/real_data/ransac_line_baseline.py : classical RANSAC line-fit
  baseline on real images (Sobel edges + RANSAC), evaluated on the
  identical held-out split train_real.py uses, for a real apples-to-
  apples comparison rather than a number reported in isolation.

## Status
Smoke tests pass end to end for all four synthetic families:
`python train.py --smoke`, `--family circle --smoke`, `--family ellipse --smoke`,
`--family line --smoke`.
Real-data path (train_real.py) verified end to end on both 'lane' and
'line' against genuine TuSimple photographs; see v23 for real numbers.

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

### v16: figures
Two kinds, handled differently -- an image generator (DALL-E-style) is the
wrong tool for either; both are built as real matplotlib/vector graphics.

- `code/make_schematic_figure.py`: conceptual dense-vs-factorized
  accumulator diagram, no data dependency, numerically matched to the
  measured Table 1 numbers (26.9x, 152KB vs 4MB). Inserted (Figure 1,
  Section 3.4).
- `code/make_figures.py`: data-driven figures from a real checkpoint.
  DONE for parabola (clean fit: anchor peak on the true vertex, sharp
  Stage-2 profile, near-pixel-tight overlay) and ellipse (a corrupted-
  setting example that happens to visually explain the robustness gap in
  Table ellipse_interim: diffuse Stage-1 accumulator, noisy multimodal
  Stage-2 profile, visibly mislocalized detections). Both sets are now in
  the paper (Figures qualitative_parabola, qualitative_ellipse).
  Still worth doing if there's time: a clean/easy ellipse example
  (`--load ellipse_easy.pt --easy`) alongside the corrupted one, so the
  paper shows the same clean-vs-corrupted contrast visually that Table
  ellipse_interim shows numerically. Note the script always writes
  `<kind>_<family>.png` regardless of which checkpoint was used, so a
  second ellipse run will silently overwrite the first -- rename the files
  in between, or add a `--tag` suffix arg before generating a second set.
  Circle figures not yet generated.

### v21: review triage -- softened claims, demoted Prop 2, two new real figures/tables
Actioned (all cheap, no training needed, everything real):
- Abstract's broadest claim scoped to "the baselines evaluated" rather than
  an unqualified "free-form learned parameter prediction" (still true, less
  sweeping).
- Proposition 2 (permutation invariance) demoted from a numbered proposition
  to a prose remark -- a reviewer correctly noted it's an immediate
  consequence of averaging over a set, not proposition-worthy.
- NEW: `code/measure_memory.py` now also measures the real ellipse (d=5)
  accumulator and computes the dense-d=5 arithmetic size (not instantiated
  -- 4GB, that's the point). Real result: dense grows 1,024x from d=3 to
  d=5 at fixed B=32 (4MB -> 4GB), factorized grows only 1.6x (152KB ->
  236KB). This is a genuinely strong, free empirical answer to "does the
  tractability argument hold as dimensionality grows" -- added as
  Table memory_scaling and referenced from the new Conclusion paragraph.
- NEW: `code/make_benchmark_examples.py` -- clean vs. corrupted example
  grid across all three families, pure generator output, no checkpoint
  needed. Inserted early in Experiments (Figure benchmark_examples) per
  reviewer request.
- Fixed a second stale "parabolas and circles" mention in the Experiments
  benchmark paragraph (missed ellipse the first time this was fixed).
- Added the 64px-vs-128px scale caveat to Limitations (was referenced but
  never actually stated).
- Added a scoping paragraph to the Conclusion: this establishes feasibility
  of factorized voting beyond lines, not state-of-the-art curve detection.

NOT actioned (real experiments the review asked for; tracked in Remaining
work below, not faked):
- Pooling ablation (max/mean/logsumexp), top-k ablation.
- A demonstration that coarse-to-fine refinement recovers ellipse's lost
  robustness -- the refinement itself isn't implemented yet (item 9).

### v22: line family -- lines as the degenerate case of the same cascade
Added `line` (theta, r direct voting, no Stage 2 at all -- exactly
recovers the original DHT when a curve family has no anchor separate
from its own parameters). Not a separate model: same encoder, same
training loop, same detection code path as every other family, just
with Stage 2 skipped. Synthetic results: F=0.910 (easy) / 0.822
(corrupted) -- the best robustness of any family, and the smallest
absolute clean-to-corrupted drop (0.088 vs parabola's 0.125 and
ellipse's 0.352). Motivated everything that follows: this is the
family used for the near-straight majority of real lane markings.

### v23: real-world validation on TuSimple, one real bug found and fixed,
### one real improvement found and shipped
Fitting real TuSimple lanes to our parabola family reveals a genuine
bimodal structure, not assumed: of 10,889 lanes with >=3 labeled
points, 3,823 (35.1%) fit a numerically stable parabola (-> 'lane'
family), 7,066 (64.9%) are near-straight enough that the fit is
unstable (-> 'line' family fallback).

Found and fixed a real coordinate-order bug in the process: self.anchors
stores (x,y) uniformly across all families, but curve_pixels' 'lane'
branch unpacked incoming params as (y0,x0,a) -- an anchor labeled
(x=A,y=B) actually rendered near true image position (x=B,y=A). The
original code had accidentally compensated for this via a matching
inversion in the ground-truth targeting; a first attempted fix patched
only the target-generation side and made real results WORSE, because it
removed the accidental cancellation without fixing the root cause.
Root-cause fix: relabeled lane to store (x0,y0,a) like every other
family (curve_pixels, sample_curve, dataset.py, tusimple_loader.py,
tusimple_dataset.py) -- no more swap_xy special-casing anywhere.
Verified via a controlled single-example overfit (sub-pixel recovery)
and full regression across all five families. Worth remembering
precisely: "just add a swap" is the natural-looking wrong fix here;
already tried it once.

Found a second, independent real-data issue: a parabola fit can be
numerically stable while its vertex still extrapolates outside the
photographed frame entirely (a gently-curving lane's true vertex need
not be in view). Since self.anchors only spans [0,size), such targets
are unlearnable in practice, not merely noisy. Fixed with a
vertex_margin filter in tusimple_dataset.py (separate from the fit's
own numerical-stability check) that drops lanes whose transformed
vertex falls too far outside the frame. Improved real 'lane' F by
roughly 2x -- and training on the resulting SMALLER filtered set still
beat training on the larger unfiltered one, confirming those examples
were actively hurting the shared encoder, not just uninformative.

Real results (held-out TuSimple split, letterboxed to 128px):
- line: F=0.50 (mean matched similarity 0.58, median 0.67) -- close to
  the synthetic corrupted-setting similarity distribution despite the
  much harder real input domain.
- lane: F=0.09-0.18 (real run-to-run variance observed at this data
  scale, ~1,500-1,800 real training images after filtering -- not yet
  resolved with repeated runs).
- RANSAC line baseline, same edge detection, same held-out split:
  F=0.253 -- our line model beats it by roughly 2x.

This reproduces the paper's central synthetic finding on real
photographs: the structurally simpler family (line, no Stage 2)
transfers far more successfully than the family requiring conditional
shape refinement (lane/parabola) -- the same dimensionality-robustness
ordering visible in the synthetic corrupted-setting results, now with
real evidence outside procedurally generated data.

### v24: paper finalized with real-data section
Added a full Real-World Validation section (the v23 findings above),
extended the line family into the synthetic-results tables, updated
abstract/contributions/limitations/conclusion accordingly. Two versions
maintained: an anonymized two-column version and a non-anonymous
single-column version (full un-trimmed prose, no page limit) for
arXiv/Zenodo/personal site.

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
10. Real-data 'lane' F is still weak and noisy across runs. Likely
    needs either more real training data (TuSimple's test_set is
    untouched and has labels -- the straightforward next source) or
    enough repeated runs to report a real mean+spread instead of a
    single number.
11. Joint multi-family real-image detection not implemented -- 'lane'
    and 'line' are currently trained and evaluated as two separate
    models on real data, even though both are already instances of the
    same cascade. Natural next architectural step.
12. Real-world validation only attempted for lane/line; circle and
    ellipse have no real-data equivalent yet (no obvious real dataset
    as directly available as TuSimple is for lane-like curves).

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
- A curve family with no natural anchor separate from its own parameters
  (line: theta, r) is handled as a degenerate case of the same cascade
  (Stage 2 skipped), not a separate model -- keeps lines and curves in one
  architecture and is what makes the real-data lane/line split coherent.
- All spatial families store params uniformly as (x0, y0, shape...); no
  per-family coordinate ordering exceptions, after v23 found out the hard
  way what that costs.
