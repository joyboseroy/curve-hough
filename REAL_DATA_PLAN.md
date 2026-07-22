# Real-data validation: TuSimple lanes

## What this is
A next step after the arXiv submission: validate the factorized cascade on
a real, standard dataset instead of only the synthetic benchmark. TuSimple
is the right first target because every baseline already cited in the paper
(PolyLaneNet, LSTR, BezierLaneNet, BSNet, HoughLaneNet) reports published
numbers on it -- a real, external comparison, not baselines we reimplemented
ourselves.

## Getting the data (not possible from the sandbox that built this repo)
TuSimple's images are a few GB and gated behind a request/mirror, not
pip-installable or on GitHub directly. On your own machine or Colab:
1. See https://github.com/TuSimple/tusimple-benchmark/issues/3 for current
   download links/mirrors (the original host has moved around over the
   years; check that issue thread for whatever's current).
2. You want `train_set.zip` (+ `test_set.zip` for held-out eval) and the
   `label_data_*.json` files.
3. Layout after extracting: `clips/<date>/<clip_id>/1.jpg ... 20.jpg`, with
   `label_data_0313.json` etc. at the top level, one JSON object per line
   (see `real_data/tusimple_loader.py` docstring for the exact schema,
   verified against TuSimple's own docs).

## What's already built (this repo)
- `hough.py`: new `lane` family, `x = a(y - y0)^2 + x0` -- the orientation
  real near-vertical lane markings actually need, as opposed to the
  synthetic `parabola` family's `y = a(x - x0)^2 + y0`. Verified this
  constructs correctly (`FactorizedDHT('lane', ...)`).
- `metrics.py`: matching `lane` branch in `sample_curve`, so curve-EA scores
  lane detections correctly (tangent direction is also swapped).
- `real_data/tusimple_loader.py`: parses TuSimple's JSON schema and
  least-squares-fits each real lane's point set to `(y0, x0, a)` in our
  vertex form. Self-tested against a hand-written mock file matching the
  real schema (recovers injected curvature to ~1e-16, correctly handles the
  `-2` missing-point sentinel) -- this validates the PARSER only, not
  real-world detection performance, since no real images have been run
  through it yet.

## What's NOT solved yet -- real gaps, not hidden
1. **Non-square images.** TuSimple frames are 1280x720; our model assumes a
   single square `size` for the anchor grid and curve rasterization.
   Short-term workaround: resize/letterbox to square before feeding the
   model (lossy, distorts curvature). Real fix: generalize `FactorizedDHT`
   to independent H and W -- genuine architecture work, not done here.
2. **Multiple lanes per image** (TuSimple has 2-5, our synthetic benchmark
   was tuned for 1-3). `topk` and eval-count assumptions need re-checking,
   not just reused as-is.
3. **Near-straight lanes.** `fit_lane_params` returns `None` when the fitted
   curvature is near zero, since vertex form is unstable there. Real
   highway lanes are very often nearly straight. Decide before training
   whether to (a) drop near-straight lanes from the real-data eval, which
   biases the comparison toward curved sections only, or (b) add a
   fallback straight-line fit and a way to mix both into one evaluation --
   this is a real design decision, not yet made.
4. **Annotation looseness.** TuSimple's own labels are documented as not
   always tightly tracking the actual marking. Expect nonzero fit residual
   even on "clean" lanes; don't read early real-data curve-EA scores as
   directly comparable to the synthetic benchmark's numbers without
   accounting for this.

## Suggested next steps, in order
1. Download a small slice of real TuSimple data and run
   `real_data/tusimple_loader.py`'s parsing against it (not the mock) --
   confirm the schema assumptions hold on genuine files before anything else.
2. Decide the near-straight-lane policy (item 3 above).
3. Pick the square-image workaround for now (item 1) and get one real
   training run going end to end, even imperfectly, before investing in the
   full non-square generalization. Note `train.py` currently only knows how
   to generate synthetic data (`SyntheticCurves`); training on real images
   needs a small new entry point (a `RealLaneDataset` wrapping
   `tusimple_loader.load_tusimple_labels` + image loading/resizing) feeding
   the same `FactorizedDHT('lane', ...)` model, not a modification to the
   synthetic generator itself. `lane` is deliberately absent from
   `train.py --family`'s choices for this reason.
4. Only then compare against the published PolyLaneNet/LSTR/BezierLaneNet/
   BSNet/HoughLaneNet numbers -- and be honest in the comparison about which
   of those numbers come from their full pipeline (multi-lane, full
   resolution, their own training budget) versus a first-pass port of ours.
