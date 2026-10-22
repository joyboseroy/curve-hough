# Open work (nothing in the paper is marked as pending)

Experiments the paper does not contain, in rough priority order. Each one, when done, replaces a sentence in Limitations.

1. Ablations never run: probe-set size (`--suite probes --values 4 8 12 16`), noise (`--suite noise --values 0.0 0.04 0.16 0.24`),
   occlusion (`--suite occlusion --values 0 1 4`), mean pooling (`--suite pool --values mean`, 3 seeds). All with `--pool max`.
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
