# Beyond Lines: Factorized Deep Hough Transform for Parametric Curves

Fast-track arXiv paper project. Generalizes the Deep Hough Transform (Zhao et
al., TPAMI 2021) from straight lines to parametric curve families via a
cascade of low-dimensional, feature-space voting stages.

## Layout
- paper/main.tex : full paper skeleton (abstract and intro drafted, method
  math complete, TODO markers for results and two citation checks)
- code/dataset.py : synthetic benchmark (parabolas, circles; noise,
  occlusion, distractor segments)
- code/hough.py : curve banks, parameter-free voting layer, FactorizedDHT
  (stage 1 marginal anchor voting + stage 2 conditional shape voting),
  stage-1 loss
- code/metrics.py : curve-EA score and Hungarian-matched P/R/F evaluation
- code/baselines.py : classical dense Hough on edges, RANSAC, direct
  regression head, DETR-lite hypothesis-query head
- code/train.py : training and evaluation entry point

## Status
Smoke test passes end to end: `python train.py --smoke`

## v4 decode fixes (after first real run showed P 0.478 / R 0.026)
- detect(): CenterNet-style 3x3 local-max NMS on the anchor map, probability
  threshold (default 0.25) instead of raw-logit > 0, soft-argmax sub-bin
  refinement of anchor coords and shape value
- stage 1: logsumexp probe pooling instead of amax
- stage 1 loss: pos_weight=8 so peak logits calibrate above threshold
- train.py: --easy flag (no noise/occlusion/distractors, single curve) and
  --thresh; eval now prints detection vs GT counts

## Validation protocol (run in this order, keep all real numbers)
1. `python train.py --family parabola --size 64 --epochs 10 --n-train 1000
   --n-test 200 --easy`  -> algorithm sanity: expect high F (0.8+). If not,
   debug before anything else.
2. Same without --easy -> the honest hard-setting number.
3. Then circles, then baselines, then 128 px on Colab for the paper table.

IMPORTANT: results tables in the paper must contain only numbers produced by
these runs. Do not paste numbers from any external draft or suggestion; no
run has produced any paper-ready table yet.

## Remaining work (ordered)
1. DONE: stage-2 teacher-forced supervision is wired (hough.py stage2_loss,
   used in train.py). Smoke F roughly doubles vs stage-1-only.
2. Full training runs: parabola and circle. Speed notes for CPU: start with
   --size 64, anchor_bins 24 (edit train.py), probe_shapes 4; the stage-1
   vote over the full anchor bank dominates cost. The stage-2 anchor bank
   cache warms up over the first epoch, so epoch 1 is the slowest. Set
   torch.set_num_threads to your core count. 128 px runs are Colab-GPU jobs.
3. Baseline runs at matched settings; fill the main results table.
4. Compute table: measure accumulator memory/FLOPs, dense 3D vs factorized.
5. Ablations: probe set size m, top-k, max vs mean pooling, refinement stage.
6. Occlusion/clutter sweep (the expected headline: voting degrades gracefully,
   regression and queries do not).
7. Verify two flagged citations in main.tex (algebraic-curve HT + CT
   application; HoughLaneNet and BSNet author lists).
8. Qualitative figure + benchmark examples figure.

## Design decisions already made
- Method core is geometric factorized voting, not learned queries. The
  transformer angle appears as (a) the discussion section: Hough voting =
  cross-attention with a geometry-fixed attention map, and (b) the DETR-lite
  baseline. Learned hypothesis spaces are future work, deliberately.
- Metric mirrors the DHT paper's EA score (contribution symmetry with the
  paper being generalized).
