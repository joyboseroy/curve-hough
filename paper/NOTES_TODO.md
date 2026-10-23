# Open work (nothing in the paper is marked as pending)

Experiments the paper does not contain, in rough priority order. Each one, when done, replaces a sentence in Limitations.

1. (Probe sweep done, see below.) Ablations never run: probe-set size (`--suite probes --values 4 8 12 16`), mean pooling (`--suite pool --values mean`, 3 seeds). All with `--pool max`.
   The k ablation is single seed; running `--suite topk --seeds 0 1 2` gives error bars.
2. Same-seed variation: seed 0 gave 0.706 (main run) and 0.690 (k ablation). Cause not traced (probably GPU nondeterminism).
   Fix by seeding cudnn deterministic, or just report 3 seeds everywhere.
3. Table 1 is one seed. Run circle and ellipse for seeds 1 and 2 with max pooling and tuned RANSAC.
4. Regression and DETR-lite for circle and ellipse.
5. Tuned classical Hough for circle.
6. Dense d=4 end-to-end comparison (the paper only claims memory at d=5).
7. TuSimple: tuned RANSAC line baseline, several seeds, validation-selected thresholds, published lane detectors with their official metric.
8. Stage-1 fix (coarse-to-fine refinement or multi-peak Stage 2), then rerun the separation sweep. This is the largest accuracy gain available (0.938 oracle vs 0.706).
9. Qualitative figures were dropped from the paper (the old ones came from log-sumexp models). Regenerate Stage-1/Stage-2/detection panels from a max-pooling checkpoint and add them back.
10. Check the two new references (Fitzgibbon et al. 1999, Hartley and Zisserman 2004). Add the acknowledgments the Zenodo copy has, if wanted.

## Removed from the paper on review

The earlier-protocol results (log-sumexp, test-swept threshold, F clean/corrupted): line 0.910/0.822, parabola 0.650/0.525, circle n.r./0.533,
ellipse 0.681/0.329. They live only in the public Zenodo version and this note. The "robustness degrades with shape dimension" argument
rested on them and is no longer made. The "this paper is a revision" paragraph was also removed; the protocol is described neutrally.
Consider a short changelog in the README for the Zenodo version history.

## Done since
Noise sweep (seed 0): F 0.683/0.700/0.706/0.696/0.700 for sigma 0/0.04/0.08/0.16/0.24, flat. Now Section 6.5 of the paper.
Occlusion sweep (seed 0, occ_max 0/1/2/4): F 0.697/0.711/0.706/0.687, flat. In Section 6.5 with the noise sweep.
Still open: probe-size ablation; mean pooling; an occlusion test by fraction of curve removed (the generator's occ_max counts rectangles, not support); a clean single-curve control under the new protocol (--easy) to test the 'Stage 1 sets the plateau' reading.

## Probe sweep (seed 0): the biggest lever found so far
m=4/6/8/12/16 -> test F 0.667/0.690/0.695/0.771/0.783, val F 0.648/0.674/0.701/0.767/0.767, anchor recall 0.719/0.786/0.788/0.855/0.867.
Next, in order: (a) parabola m=12 and 16 over seeds 0 1 2 (`--suite probes --values 12 16 --seeds 0 1 2 --pool max`),
(b) circle and ellipse with more probes (ellipse uses `--ellipse-probe-per-axis`, default 3 = 27; try 4 = 64),
(c) rerun the whole of Table 1 at the validation-selected m, with the tuned RANSAC rows unchanged, then update the abstract numbers,
(d) maybe m=24 or 32 to find where it saturates, and time Stage 1 (compute grows linearly in m).
If (a) holds, the paper's headline changes from "ties RANSAC" to "beats tuned RANSAC on parabolas", so do not edit that claim before (a).

## Selection rule, recorded before the 3-seed run
Pick the smallest m that attains the maximum validation F in the probe sweep (seed 0: m=12, val F 0.767, tied with m=16). Run m=12 on seeds 0 1 2:
`python ablate.py --suite probes --values 12 --family parabola --size 64 --epochs 10 --n-train 1000 --n-val 200 --n-test 200 --seeds 0 1 2 --pool max --out abl_max_probe12 --resume`
If the 3-seed mean is about 0.75 or more, make m=12 the main parabola setting (keep m=6 as the pre-specified baseline row), and rerun circle and ellipse at their own validation-selected m. If it falls to about 0.71, report it as variance and leave the headline as is.

## TuSimple rerun (code ready, results not yet in the paper)
Command in REPRODUCE.md ("TuSimple under the v24 protocol"). It gives, over 3 seeds: line and lane models (max pool, validation threshold),
tuned and untuned RANSAC for both families, and lane with log-sumexp for comparison. When the table is back, rewrite Section 7 and
Table 9 (drop "single run", "old protocol", "untuned RANSAC" caveats that no longer apply; keep: two separate models, random frame split,
no published detectors). Still not done afterwards: published lane detectors under the official TuSimple metric.

## Update: 3-seed m=12 and 3-seed RANSAC (parabola)
- m=12 cascade 0.753 +/- 0.013; m=6 0.696 +/- 0.011; RANSAC tuned 0.725 +/- 0.016; untuned 0.566 +/- 0.011. Paper updated (Table 2, abstract, limitations, conclusion).
- Open: circle and ellipse over 3 seeds and with more probes; circle RANSAC seeds 0-2 pending.

## Update: TuSimple done (3 seeds, v24 protocol)
Paper Section 7 and Table 9 rewritten. Open: extend threshold grid below 0.05 for lanes, clip-level split, road-mask RANSAC baseline, published lane detectors, single model for both regimes, lane Stage-1 recall (0.15) improvement.
