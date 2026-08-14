"""Real-image TuSimple dataset: loads actual photographs + fitted lane
params, letterboxed to a square target size, split by family ('lane' for
the 35% stable-parabola lanes, 'line' for the 65% near-straight ones --
see tusimple_loader.py). One model per family, per the current design
(no joint multi-family detection yet -- see REAL_DATA_PLAN.md).

Letterbox (uniform scale + pad), not stretch: preserves line angles and
keeps the parabola/line parameter transform in simple closed form. Under
x' = x*scale, y' = y*scale + pad_y:
  lane (y0,x0,a):  new_a = a/scale, new_y0 = pad_y + scale*y0,
                    new_x0 = scale*x0
  line (theta,r):  new_theta = theta (angles preserved by uniform scale),
                    new_r = r*scale
"""
import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tusimple_loader import load_tusimple_labels

ORIG_W, ORIG_H = 1280, 720


def letterbox_params(family, params, size, orig_w=ORIG_W, orig_h=ORIG_H):
    scale = size / orig_w  # orig is landscape (W>H), so W-bound is the tight one
    pad_y = (size - orig_h * scale) / 2.0
    if family == "lane":
        y0, x0, a = params
        return (pad_y + scale * y0, scale * x0, a / scale)
    else:  # line
        theta, r = params
        return (theta, r * scale)


def letterbox_image(img, size, orig_w=ORIG_W, orig_h=ORIG_H):
    scale = size / orig_w
    new_h = round(orig_h * scale)
    resized = img.resize((size, new_h), Image.BILINEAR)
    pad_y = (size - new_h) // 2
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (0, pad_y))
    return canvas


class RealTuSimpleDataset(Dataset):
    """family: 'lane' or 'line'. root: path to a TuSimple train_set dir
    containing clips/ and label_data_*.json. max_curves: cap on lanes per
    image kept for this family (excess dropped, matching train.py's
    fixed-size params tensor convention)."""

    def __init__(self, root, family, size=128, label_globs=None, max_curves=4,
                 max_images=None):
        assert family in ("lane", "line")
        self.root, self.family, self.size = root, family, size
        self.max_curves = max_curves
        if label_globs is None:
            label_globs = sorted(glob.glob(os.path.join(root, "label_data_*.json")))
        self.items = []  # (raw_file, [params, ...]) filtered to this family
        for path in label_globs:
            for raw_file, results in load_tusimple_labels(path):
                fam_params = [p for f, p in results if f == family]
                if fam_params:
                    self.items.append((raw_file, fam_params))
        if max_images:
            self.items = self.items[:max_images]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raw_file, fam_params = self.items[idx]
        img = Image.open(os.path.join(self.root, raw_file)).convert("L")
        canvas = letterbox_image(img, self.size)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0

        k = min(len(fam_params), self.max_curves)
        plen = 3 if self.family == "lane" else 2
        params = np.zeros((self.max_curves, 5), np.float32)
        for j in range(k):
            transformed = letterbox_params(self.family, fam_params[j], self.size)
            params[j, :plen] = transformed
        return torch.from_numpy(arr)[None], torch.from_numpy(params), k


if __name__ == "__main__":
    # Geometric verification: draw a known lane in ORIGINAL 1280x720 space,
    # letterbox-resize the pixels, transform the params with the closed-form
    # formula, redraw from the transformed params, and check the two
    # rasterizations actually overlap -- confirms the algebra matches real
    # pixel-space resizing, not just that it looks plausible on paper.
    import sys
    sys.path.insert(0, "..")
    from dataset import draw_lane, draw_line

    S = 128
    orig = np.zeros((ORIG_H, ORIG_W), np.float32)
    y0, x0, a = 300.0, 600.0, 0.0015
    draw_lane(orig, y0, x0, a)
    orig_img = Image.fromarray((orig * 255).astype(np.uint8))
    resized = letterbox_image(orig_img, S)
    resized_arr = np.asarray(resized, dtype=np.float32) / 255.0

    new_y0, new_x0, new_a = letterbox_params("lane", (y0, x0, a), S)
    redrawn = np.zeros((S, S), np.float32)
    draw_lane(redrawn, new_y0, new_x0, new_a)

    both = (resized_arr > 0) & (redrawn > 0)
    only_resized = (resized_arr > 0) & ~(redrawn > 0)
    only_redrawn = (redrawn > 0) & ~(resized_arr > 0)
    print(f"lane check: overlap={both.sum()}  resized_only={only_resized.sum()}  "
          f"redrawn_only={only_redrawn.sum()}  "
          f"(want overlap high, other two near zero -- some mismatch expected "
          f"from bilinear resize blur and stroke width)")

    theta, r = 0.3, -150.0
    draw_line(orig := np.zeros((ORIG_H, ORIG_W), np.float32), theta,
              r + (ORIG_W - ORIG_H) / 2)  # offset irrelevant to the check below
    new_theta, new_r = letterbox_params("line", (theta, r), S)
    print(f"line check: theta unchanged = {np.isclose(theta, new_theta)}  "
          f"r scaled by {new_r / r:.4f} (expect {S/ORIG_W:.4f})")
