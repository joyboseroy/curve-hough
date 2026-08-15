"""Factorized Deep Hough Transform for parametric curves.

Stage 1: marginal anchor voting over (x0, y0), pooling over a coarse shape
probe set. Stage 2: conditional dense shape voting at top-k anchor peaks.
Voting layers are parameter-free gather-and-mean operations, so gradients
flow to the encoder exactly as in DHT.

The 'line' family is a special case: a line has no spatial anchor separate
from its own parameters (theta, r fully describe it), so it reuses Stage 1's
single-pass voting directly over (theta, r) space and skips Stage 2 --
this is the original DHT's line detection, folded into the same class.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ---------------------------------------------------------------- curve banks
def curve_pixels(family, p, H, W, n=400):
    """Flattened pixel indices of curve C(p) clipped to the image.
    p is (x0, y0, *shape) for spatial families: parabola shape=(a,),
    circle shape=(r,), ellipse shape=(rx, ry, phi), lane shape=(a,)
    (near-vertical, x = a(y-y0)^2 + x0). 'line' is p = (theta, r) directly,
    no spatial anchor."""
    if family == "parabola":
        x0, y0, a = p[0], p[1], p[2]
        xs = np.linspace(0, W - 1, n)
        ys = a * (xs - x0) ** 2 + y0
    elif family == "lane":
        x0, y0, a = p[0], p[1], p[2]
        ys = np.linspace(0, H - 1, n)
        xs = a * (ys - y0) ** 2 + x0
    elif family == "circle":
        x0, y0, r = p[0], p[1], p[2]
        t = np.linspace(0, 2 * np.pi, n)
        xs, ys = x0 + r * np.cos(t), y0 + r * np.sin(t)
    elif family == "ellipse":
        x0, y0, rx, ry, phi = p[0], p[1], p[2], p[3], p[4]
        t = np.linspace(0, 2 * np.pi, n)
        ct, st = np.cos(phi), np.sin(phi)
        xs = x0 + rx * np.cos(t) * ct - ry * np.sin(t) * st
        ys = y0 + rx * np.cos(t) * st + ry * np.sin(t) * ct
    elif family == "line":
        theta, r = p[0], p[1]
        # x*cos(theta) + y*sin(theta) = r, origin at image center
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        ct, st = np.cos(theta), np.sin(theta)
        if abs(st) > abs(ct):
            xs = np.linspace(0, W - 1, n)
            ys = (r - (xs - cx) * ct) / st + cy
        else:
            ys = np.linspace(0, H - 1, n)
            xs = (r - (ys - cy) * st) / ct + cx
    else:
        raise ValueError(f"unknown family {family}")
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xi = np.clip(np.round(xs[m]).astype(np.int64), 0, W - 1)
    yi = np.clip(np.round(ys[m]).astype(np.int64), 0, H - 1)
    idx = np.unique(yi * W + xi)
    return idx


def build_bank(family, anchors, shapes, H, W, max_pts=256):
    """Index bank [A, S, max_pts] (padded with -1) for anchors x shapes.
    For 'line', 'anchors' are actually the full (theta, r) points and
    'shapes' is a single dummy [()] so the [A, S, P] shape stays uniform."""
    A, S = len(anchors), len(shapes)
    bank = np.full((A, S, max_pts), -1, np.int64)
    for i, a in enumerate(anchors):
        for j, s in enumerate(shapes):
            idx = curve_pixels(family, (*a, *s), H, W)
            if len(idx) > max_pts:
                idx = idx[np.linspace(0, len(idx) - 1, max_pts).astype(int)]
            bank[i, j, :len(idx)] = idx
    return torch.from_numpy(bank)


def stage1_accumulate(feat, bank1, num_anchors, pool="logsumexp"):
    """Chunked stage-1 voting: loop over probe shapes one at a time instead
    of gathering all probes simultaneously, bounding peak memory to one
    probe's worth. For 'line' (S=1 dummy shape) this is just one pass."""
    B, C, N = feat.shape
    S = bank1.shape[1]
    slices = []
    for s in range(S):
        slices.append(vote(feat, bank1[:, s, :]))       # [B, C, A]
    stacked = torch.stack(slices, dim=-1)                # [B, C, A, S]
    if pool == "logsumexp":
        pooled = torch.logsumexp(stacked, dim=-1)
    else:
        pooled = stacked.mean(-1)
    return pooled.reshape(B, C, num_anchors, num_anchors) if isinstance(num_anchors, int) \
        else pooled.reshape(B, C, *num_anchors)


def vote(feat, bank):
    """feat: [B, C, H*W]; bank: [..., P] indices with -1 padding.
    Returns mean feature along each curve: [B, C, *bank.shape[:-1]]."""
    B, C, N = feat.shape
    shape = bank.shape
    flat = bank.reshape(-1)
    valid = (flat >= 0).float()
    idx = flat.clamp(min=0)
    g = feat[:, :, idx]
    g = g * valid
    g = g.reshape(B, C, -1, shape[-1])
    cnt = valid.reshape(-1, shape[-1]).sum(-1).clamp(min=1)
    out = g.sum(-1) / cnt
    return out.reshape(B, C, *shape[:-1])


# ------------------------------------------------------------------- encoder
class SmallEncoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


LINE_FAMILY = "line"


# ------------------------------------------------------- cascaded DHT detector
class FactorizedDHT(nn.Module):
    def __init__(self, family="parabola", size=128, ch=32,
                 anchor_bins=32, probe_shapes=6, dense_shapes=48, topk=4,
                 ellipse_probe_per_axis=3, ellipse_dense_per_axis=6,
                 bank_max_pts=128, bank2_cache_size=64,
                 theta_bins=90, r_bins=90, shape_range=(0.004, 0.06)):
        super().__init__()
        self.family, self.size, self.topk = family, size, topk
        self.bank_max_pts = bank_max_pts
        self.bank2_cache_size = bank2_cache_size
        self.is_line = (family == LINE_FAMILY)

        if self.is_line:
            # A line has no spatial anchor separate from its own params:
            # (theta, r) IS the anchor. Single-stage voting, no Stage 2.
            self.Ba_theta, self.Ba_r = theta_bins, r_bins
            thetas = np.linspace(0, np.pi, theta_bins, endpoint=False)
            diag = np.sqrt(size ** 2 + size ** 2) / 2
            rs = np.linspace(-diag, diag, r_bins)
            self.anchors = [(t, r) for t in thetas for r in rs]
            self.probe = [()]   # no shape dimension at all
            self.dense = [()]
            self.shape_dim = 0
            self.register_buffer("bank1", build_bank(family, self.anchors, self.probe,
                                                       size, size, max_pts=bank_max_pts))
            self.encoder = SmallEncoder(ch)
            self.head1 = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, 1, 1))
            self._bank2_cache = OrderedDict()   # unused for line, kept for API parity
            return

        self.Ba = anchor_bins
        ax = np.linspace(0, size - 1, anchor_bins)
        self.anchors = [(x, y) for y in ax for x in ax]  # row-major (y outer)
        if family in ("parabola", "lane"):
            lo, hi = shape_range
            mag = np.geomspace(lo, hi, probe_shapes // 2)
            probe_vals = np.concatenate([-mag[::-1], mag])
            dm = np.geomspace(lo, hi, dense_shapes // 2)
            dense_vals = np.concatenate([-dm[::-1], dm])
            self.probe = [(v,) for v in probe_vals]
            self.dense = [(v,) for v in dense_vals]
        elif family == "circle":
            probe_vals = np.linspace(0.08 * size, 0.35 * size, probe_shapes)
            dense_vals = np.linspace(0.08 * size, 0.35 * size, dense_shapes)
            self.probe = [(v,) for v in probe_vals]
            self.dense = [(v,) for v in dense_vals]
        else:  # ellipse
            p = ellipse_probe_per_axis
            rx_p = np.linspace(0.10 * size, 0.30 * size, p)
            ry_p = np.linspace(0.06 * size, 0.22 * size, p)
            phi_p = np.linspace(0, np.pi, p, endpoint=False)
            self.probe = [(rx, ry, phi) for rx in rx_p for ry in ry_p for phi in phi_p]
            d = ellipse_dense_per_axis
            rx_d = np.linspace(0.10 * size, 0.30 * size, d)
            ry_d = np.linspace(0.06 * size, 0.22 * size, d)
            phi_d = np.linspace(0, np.pi, d, endpoint=False)
            self.dense = [(rx, ry, phi) for rx in rx_d for ry in ry_d for phi in phi_d]
        self.shape_dim = len(self.dense[0])
        self._dense_arr = np.array(self.dense)
        self._dense_range = self._dense_arr.max(0) - self._dense_arr.min(0) + 1e-6
        self.register_buffer("bank1", build_bank(family, self.anchors, self.probe,
                                                   size, size, max_pts=bank_max_pts))
        self.encoder = SmallEncoder(ch)
        self.head1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1))
        self.head2 = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, 1, 1))
        self._bank2_cache = OrderedDict()

    # ---------------------------------------------------------- line path
    def _forward_line(self, img):
        B = img.shape[0]
        X = self.encoder(img)
        feat = X.flatten(2)
        Y1 = stage1_accumulate(feat, self.bank1, (self.Ba_theta, self.Ba_r))
        P1 = self.head1(Y1)                                  # [B,1,Btheta,Br]
        return P1, feat

    @torch.no_grad()
    def _detect_line(self, img, thresh=0.25):
        P1, feat = self._forward_line(img)
        B = img.shape[0]
        prob = torch.sigmoid(P1)
        pooled = F.max_pool2d(P1, 3, stride=1, padding=1)
        peaks = (P1 == pooled).float() * prob
        out = []
        for b in range(B):
            pk = peaks[b, 0].flatten()
            vals, idxs = pk.topk(self.topk)
            dets = []
            for v, a_idx in zip(vals.tolist(), idxs.tolist()):
                if v < thresh:
                    continue
                theta, r = self.anchors[a_idx]
                dets.append((theta, r, v))
            out.append(dets)
        return out

    # --------------------------------------------------------- shared bank2
    def bank2_for(self, a_idx, device):
        if a_idx in self._bank2_cache:
            self._bank2_cache.move_to_end(a_idx)
            return self._bank2_cache[a_idx].to(device)
        a = self.anchors[a_idx]
        b = build_bank(self.family, [a], self.dense, self.size, self.size,
                       max_pts=self.bank_max_pts)[0]
        self._bank2_cache[a_idx] = b
        if len(self._bank2_cache) > self.bank2_cache_size:
            self._bank2_cache.popitem(last=False)
        return b.to(device)

    def anchor_index(self, x, y):
        scale = (self.Ba - 1) / (self.size - 1)
        xi = int(round(float(x) * scale))
        yi = int(round(float(y) * scale))
        xi = min(max(xi, 0), self.Ba - 1)
        yi = min(max(yi, 0), self.Ba - 1)
        return yi * self.Ba + xi

    def stage2_loss(self, feat, params, counts, sigma=1.0):
        if self.is_line:
            return feat.new_zeros(())
        dense_norm = self._dense_arr / self._dense_range
        total, n = feat.new_zeros(()), 0
        for b in range(feat.shape[0]):
            for j in range(counts[b]):
                p = params[b][j]
                a_idx = self.anchor_index(p[0], p[1])
                bank2 = self.bank2_for(a_idx, feat.device)
                y2 = vote(feat[b:b + 1], bank2)
                logits = self.head2(y2)[0, 0]
                gt_shape = p[2:2 + self.shape_dim].cpu().numpy() / self._dense_range
                s_bin = int(np.argmin(((dense_norm - gt_shape) ** 2).sum(-1)))
                idxs = torch.arange(len(self.dense), device=feat.device, dtype=torch.float32)
                tgt = torch.exp(-((idxs - s_bin) ** 2) / (2 * sigma ** 2))
                total = total + F.binary_cross_entropy_with_logits(logits, tgt)
                n += 1
        return total / max(n, 1)

    def stage1_target(self, params_batch, counts, sigma=1.5):
        """Ground-truth target map for Stage 1, dispatched by family.
        Spatial families: params are (x0, y0, ...), grid is size x size in
        pixel space. Line: params are (theta, r) directly (a line has no
        separate spatial anchor), grid is theta_bins x r_bins in the
        family's own (theta, r) index space -- reusing the spatial
        gaussian_target here would silently produce a wrong target, since
        neither the coordinate scale nor the grid shape match."""
        B = len(params_batch)
        if self.is_line:
            thetas = np.linspace(0, np.pi, self.Ba_theta, endpoint=False)
            diag = np.sqrt(self.size ** 2 + self.size ** 2) / 2
            rs = np.linspace(-diag, diag, self.Ba_r)
            d_theta = thetas[1] - thetas[0]
            d_r = rs[1] - rs[0]
            targets = []
            for b in range(B):
                t = torch.zeros(self.Ba_theta, self.Ba_r)
                tt, rr = torch.meshgrid(torch.arange(self.Ba_theta),
                                        torch.arange(self.Ba_r), indexing="ij")
                for j in range(counts[b]):
                    gtheta, gr = float(params_batch[b][j][0]), float(params_batch[b][j][1])
                    ci = gtheta / d_theta
                    cj = (gr - rs[0]) / d_r
                    t = torch.maximum(t, torch.exp(
                        -((tt - ci) ** 2 + (rr - cj) ** 2) / (2 * sigma ** 2)))
                targets.append(t)
            return torch.stack(targets)[:, None]
        return torch.stack([
            gaussian_target(self.Ba, [(p[0], p[1]) for p in params_batch[b][:counts[b]]],
                            self.size, sigma=sigma)
            for b in range(B)])[:, None]

    def stage1_loss(self, P1, params_batch, counts):
        tgt = self.stage1_target(params_batch, counts).to(P1.device)
        return F.binary_cross_entropy_with_logits(
            P1, tgt, pos_weight=torch.tensor(8.0, device=P1.device))

    def forward(self, img):
        if self.is_line:
            P1, feat = self._forward_line(img)
            return P1, None, None, feat
        B = img.shape[0]
        X = self.encoder(img)
        feat = X.flatten(2)
        Y1 = stage1_accumulate(feat, self.bank1, self.Ba)
        P1 = self.head1(Y1)
        flat = P1.flatten(1)
        top = flat.topk(self.topk, dim=1).indices
        P2, picked = [], []
        for b in range(B):
            row = []
            for a_idx in top[b].tolist():
                bank2 = self.bank2_for(a_idx, img.device)
                y2 = vote(feat[b:b + 1], bank2)
                row.append(self.head2(y2))
            P2.append(torch.cat(row, 0))
            picked.append(top[b])
        return P1, torch.stack(P2), torch.stack(picked), feat

    @torch.no_grad()
    def detect(self, img, thresh=0.25):
        if self.is_line:
            return self._detect_line(img, thresh=thresh)
        B = img.shape[0]
        X = self.encoder(img)
        feat = X.flatten(2)
        Y1 = stage1_accumulate(feat, self.bank1, self.Ba)
        P1 = self.head1(Y1)
        pooled = F.max_pool2d(P1, 3, stride=1, padding=1)
        peaks = (P1 == pooled).float() * torch.sigmoid(P1)
        scale = (self.size - 1) / (self.Ba - 1)
        out = []
        for b in range(B):
            grid = P1[b, 0]
            pk = peaks[b, 0].flatten()
            vals, idxs = pk.topk(self.topk)
            dets = []
            for v, a_idx in zip(vals.tolist(), idxs.tolist()):
                if v < thresh:
                    continue
                ri, ci = a_idx // self.Ba, a_idx % self.Ba
                y0n, y1n = max(0, ri - 1), min(self.Ba, ri + 2)
                x0n, x1n = max(0, ci - 1), min(self.Ba, ci + 2)
                nb = grid[y0n:y1n, x0n:x1n]
                w = torch.softmax(nb.flatten(), 0)
                yy, xx = torch.meshgrid(
                    torch.arange(y0n, y1n, device=img.device, dtype=torch.float32),
                    torch.arange(x0n, x1n, device=img.device, dtype=torch.float32),
                    indexing="ij")
                ay = float((yy.flatten() * w).sum()) * scale
                ax = float((xx.flatten() * w).sum()) * scale
                bank2 = self.bank2_for(a_idx, img.device)
                y2 = vote(feat[b:b + 1], bank2)
                s_prof = self.head2(y2)[0, 0]
                s_idx = int(s_prof.argmax())
                lo, hi = max(0, s_idx - 1), min(len(self.dense), s_idx + 2)
                sw = torch.softmax(s_prof[lo:hi], 0)
                dvals = torch.as_tensor(self._dense_arr[lo:hi], dtype=torch.float32,
                                        device=img.device)
                s_val = (sw[:, None] * dvals).sum(0)
                dets.append((ax, ay, *s_val.tolist(), v))
            out.append(dets)
        return out


def gaussian_target(anchor_bins, gts, size, sigma=1.5):
    t = torch.zeros(anchor_bins, anchor_bins)
    scale = (anchor_bins - 1) / (size - 1)
    yy, xx = torch.meshgrid(torch.arange(anchor_bins), torch.arange(anchor_bins),
                            indexing="ij")
    for (gx, gy) in gts:
        cx, cy = gx * scale, gy * scale
        t = torch.maximum(t, torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return t


def stage1_loss(P1, params_batch, counts, size):
    B = P1.shape[0]
    Ba = P1.shape[-1]
    tgt = torch.stack([
        gaussian_target(Ba, [(p[0], p[1]) for p in params_batch[b][:counts[b]]], size)
        for b in range(B)]).to(P1.device)[:, None]
    return F.binary_cross_entropy_with_logits(
        P1, tgt, pos_weight=torch.tensor(8.0, device=P1.device))
