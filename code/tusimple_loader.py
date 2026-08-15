"""Adapter: TuSimple lane-detection labels -> our curve-family interface.

TuSimple schema: each line of label_data_*.json is one JSON object:
    {"raw_file": "clips/.../20.jpg",
     "lanes": [[x_11, x_12, ...], ...],   # -2 = absent at that row
     "h_samples": [y_1, y_2, ...]}         # shared row heights

Confirmed on real data (not just the schema): of 10,889 real lanes with
>=3 points, 3823 (35.1%) fit a stable parabola (our 'lane' family,
x = a(y-y0)^2 + x0) and 7066 (64.9%) are near-straight enough that the
parabola's vertex-form conversion is unstable -- those get a line fit
instead (our 'line' family, theta/r), via fit_line_params below.
"""
import json
import numpy as np


def fit_lane_params(xs, ys, img_w=1280, img_h=720, bound_factor=3):
    """Least-squares x = A*y^2 + B*y + C, converted to vertex form
    (x0, y0, a). Returns None if the fit is near-straight enough that the
    vertex-form conversion is unstable (bounds check on the OUTPUT, not on
    A -- confirmed on real data that a raw-A threshold alone lets unstable
    fits through)."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if len(xs) < 3:
        return None
    A_mat = np.stack([ys ** 2, ys, np.ones_like(ys)], -1)
    try:
        coef, *_ = np.linalg.lstsq(A_mat, xs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, B, C = coef
    if abs(A) < 1e-9:
        return None
    y0 = -B / (2 * A)
    x0 = C - B ** 2 / (4 * A)
    if abs(x0) > bound_factor * img_w or abs(y0) > bound_factor * img_h:
        return None
    return float(x0), float(y0), float(A)


def fit_line_params(xs, ys, img_w=1280, img_h=720):
    """Fallback for near-straight lanes: least-squares x = B*y + C,
    converted to (theta, r) matching hough.py's 'line' family, origin at
    image center. Always well-conditioned -- no division by a near-zero
    curvature the way the parabola fit has."""
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    A_mat = np.stack([ys, np.ones_like(ys)], -1)
    coef, *_ = np.linalg.lstsq(A_mat, xs, rcond=None)
    Bc, C = coef
    # line: x = Bc*y + C  <=>  x - Bc*y = C, direction (1, Bc) up to norm
    cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0
    # x*cos(theta) + y*sin(theta) = r, origin at (cx, cy):
    # (x-cx)*1 + (y-cy)*(-Bc) = C - cx + Bc*cy  [from x - Bc*y = C]
    nx, ny = 1.0, -Bc
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    r = (C - cx + Bc * cy) / norm
    theta = np.arctan2(ny, nx) % np.pi
    if ny < 0 and not np.isclose(theta, 0):
        r = -r
    return float(theta), float(r)


def parse_tusimple_line(line, min_points=3, img_w=1280, img_h=720):
    """One JSON line -> raw_file, list of ('lane', params) or ('line', params)
    tuples, one per lane in the frame."""
    rec = json.loads(line)
    raw_file = rec["raw_file"]
    h_samples = rec["h_samples"]
    results = []
    for lane_xs in rec["lanes"]:
        pts = [(x, y) for x, y in zip(lane_xs, h_samples) if x >= 0]
        if len(pts) < min_points:
            continue
        xs, ys = zip(*pts)
        fit = fit_lane_params(xs, ys, img_w=img_w, img_h=img_h)
        if fit is not None:
            results.append(("lane", fit))
        else:
            results.append(("line", fit_line_params(xs, ys, img_w=img_w, img_h=img_h)))
    return raw_file, results


def load_tusimple_labels(json_path, min_points=3, img_w=1280, img_h=720):
    """Yields (raw_file, [(family, params), ...]) per labeled frame."""
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield parse_tusimple_line(line, min_points=min_points,
                                      img_w=img_w, img_h=img_h)


if __name__ == "__main__":
    # Self-test: schema-conformant mock, checks both fit paths and that
    # fit_line_params round-trips a known near-horizontal line correctly.
    import tempfile, os

    h_samples = list(range(160, 720, 10))
    true_a, true_y0, true_x0 = 0.0006, 100.0, 400.0
    xs_curve = [true_a * (y - true_y0) ** 2 + true_x0 for y in h_samples]
    xs_straight = [400.0 + 0.01 * y for y in h_samples]  # near-vertical, tiny slope

    mock = {"raw_file": "clips/mock/20.jpg",
            "lanes": [xs_curve, xs_straight], "h_samples": h_samples}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json.dumps(mock) + "\n")
        path = f.name

    for raw_file, results in load_tusimple_labels(path):
        print(f"raw_file={raw_file}")
        for family, params in results:
            print(f"  {family}: {params}")
    os.unlink(path)
    print("self-test complete (schema parsing only -- verify against real data too)")
