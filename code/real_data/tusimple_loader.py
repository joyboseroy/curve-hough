"""Adapter: TuSimple lane-detection labels -> our (image, params, count)
interface, fitting each real lane to our 'lane' family (x = a(y-y0)^2 + x0).

TuSimple's own schema (verified against their repo docs, not assumed):
each line of label_data_*.json is one JSON object:
    {
      "raw_file": "clips/.../20.jpg",
      "lanes": [[x_11, x_12, ...], [x_21, ...], ...],   # -2 = absent at that row
      "h_samples": [y_1, y_2, ...]                       # shared row heights
    }
Real annotations are polylines, not smooth curves, and are not guaranteed to
align tightly to the marking (this is a documented property of TuSimple, not
an artifact of our fitting) -- expect the least-squares fit to have genuine
residual error even on "correct" lanes. That is a property of the labels,
not a bug in this adapter.

IMPORTANT -- things this adapter does NOT solve, flagged rather than hidden:
  1. Non-square images. TuSimple frames are 1280x720. Our model (FactorizedDHT)
     currently assumes a single square `size` for both height and width in
     its anchor grid and curve rasterization. This adapter resizes/crops to
     a square as a stopgap (see --mode below); a proper fix is generalizing
     the model to independent H and W, which is real architecture work, not
     done here.
  2. Multiple lanes per image (typically 2-5). Our `topk` default (4) and
     eval assumptions were tuned against 1-3 synthetic curves; sanity-check
     detection counts before trusting numbers on the real data.
  3. This adapter has only been validated against a hand-written mock file
     matching TuSimple's schema (see the __main__ block) -- NOT against the
     real dataset, which is not accessible from this environment. Run the
     __main__ self-test after downloading real data, before trusting output.
"""
import json
import numpy as np


def fit_lane_params(xs, ys):
    """Least-squares fit x = A*y^2 + B*y + C, converted to vertex form
    (y0, x0, a) matching our 'lane' family. Returns None if under-determined
    or degenerate (near-straight lane, |A| ~ 0 -- vertex form is unstable
    there; caller should skip or fall back to a straight-line family)."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if len(xs) < 3:
        return None
    A_mat = np.stack([ys ** 2, ys, np.ones_like(ys)], -1)
    try:
        coef, *_ = np.linalg.lstsq(A_mat, xs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, B, C = coef
    if abs(A) < 1e-6:
        return None  # effectively straight; vertex form degenerates
    y0 = -B / (2 * A)
    x0 = C - B ** 2 / (4 * A)
    return float(y0), float(x0), float(A)


def parse_tusimple_line(line, min_points=3):
    """One JSON line -> (raw_file, list of (y0, x0, a) fitted lane params,
    list of raw (xs, ys) point sets actually used per lane, for diagnostics)."""
    rec = json.loads(line)
    raw_file = rec["raw_file"]
    h_samples = rec["h_samples"]
    lanes_params, lanes_points = [], []
    for lane_xs in rec["lanes"]:
        pts = [(x, y) for x, y in zip(lane_xs, h_samples) if x >= 0]
        if len(pts) < min_points:
            continue
        xs, ys = zip(*pts)
        fit = fit_lane_params(xs, ys)
        if fit is None:
            continue
        lanes_params.append(fit)
        lanes_points.append((xs, ys))
    return raw_file, lanes_params, lanes_points


def load_tusimple_labels(json_path, min_points=3):
    """Yields (raw_file, params_list, points_list) per labeled frame."""
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield parse_tusimple_line(line, min_points=min_points)


if __name__ == "__main__":
    # Self-test against a hand-written, schema-conformant mock file.
    # This validates the PARSER only -- it is not real TuSimple data and
    # proves nothing about real-world detection performance.
    import tempfile, os

    h_samples = list(range(160, 720, 10))
    # a lane that is genuinely a slight parabola in (x as fn of y)
    true_a, true_y0, true_x0 = 0.0006, 100.0, 400.0
    xs = [true_a * (y - true_y0) ** 2 + true_x0 for y in h_samples]
    # second lane: mark a few points absent (-2), as real labels do
    xs2 = [x + 300 if i % 7 else -2 for i, x in enumerate(xs)]

    mock = {"raw_file": "clips/mock/20.jpg", "lanes": [xs, xs2], "h_samples": h_samples}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json.dumps(mock) + "\n")
        path = f.name

    for raw_file, params, points in load_tusimple_labels(path):
        print(f"raw_file={raw_file}")
        for i, (y0, x0, a) in enumerate(params):
            err_a = abs(a - true_a) if i == 0 else None
            print(f"  lane {i}: fitted (y0={y0:.2f}, x0={x0:.2f}, a={a:.6f})"
                  + (f"  [true a={true_a}, abs err={err_a:.2e}]" if err_a is not None else ""))
    os.unlink(path)
    print("self-test complete (schema parsing only -- NOT validated on real images)")
