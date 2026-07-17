"""Factorized Deep Hough Transform for parametric curves.

Stage 1: marginal anchor voting over (x0, y0), pooling over a coarse shape
probe set. Stage 2: conditional dense 1D shape voting at top-k anchor peaks.
Voting layers are parameter-free gather-and-mean operations, so gradients
flow to the encoder exactly as in DHT.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- curve banks
def curve_pixels(family, p, H, W, n=400):
    """Flattened pixel indices of curve C(p) clipped to the image."""
    if family == "parabola":
        x0, y0, a = p
        xs = np.linspace(0, W - 1, n)
        ys = a * (xs - x0) ** 2 + y0
    else:  # circle
        x0, y0, r = p
        t = np.linspace(0, 2 * np.pi, n)
        xs, ys = x0 + r * np.cos(t), y0 + r * np.sin(t)
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xi = np.clip(np.round(xs[m]).astype(np.int64), 0, W - 1)
    yi = np.clip(np.round(ys[m]).astype(np.int64), 0, H - 1)
    idx = np.unique(yi * W + xi)
    return idx


def build_bank(family, anchors, shapes, H, W, max_pts=256):
    """Index bank [A, S, max_pts] (padded with -1) for anchors x shapes."""
    A, S = len(anchors), len(shapes)
    bank = np.full((A, S, max_pts), -1, np.int64)
    for i, a in enumerate(anchors):
        for j, s in enumerate(shapes):
            idx = curve_pixels(family, (*a, s), H, W)
            if len(idx) > max_pts:
                idx = idx[np.linspace(0, len(idx) - 1, max_pts).astype(int)]
            bank[i, j, :len(idx)] = idx
    return torch.from_numpy(bank)


def vote(feat, bank):
    """feat: [B, C, H*W]; bank: [..., P] indices with -1 padding.
    Returns mean feature along each curve: [B, C, *bank.shape[:-1]]."""
    B, C, N = feat.shape
    shape = bank.shape
    flat = bank.reshape(-1)                       # [K*P]
    valid = (flat >= 0).float()                   # [K*P]
    idx = flat.clamp(min=0)
    g = feat[:, :, idx]                           # [B, C, K*P]
    g = g * valid
    g = g.reshape(B, C, -1, shape[-1])            # [B, C, K, P]
    cnt = valid.reshape(-1, shape[-1]).sum(-1).clamp(min=1)  # [K]
    out = g.sum(-1) / cnt                         # [B, C, K]
    return out.reshape(B, C, *shape[:-1])


# ------------------------------------------------------------------- encoder
class SmallEncoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1), nn.BatchNorm2d(ch), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------- cascaded DHT detector
class FactorizedDHT(nn.Module):
    def __init__(self, family="parabola", size=128, ch=32,
                 anchor_bins=32, probe_shapes=6, dense_shapes=48, topk=4):
        super().__init__()
        self.family, self.size, self.topk = family, size, topk
        self.Ba = anchor_bins
        ax = np.linspace(0, size - 1, anchor_bins)
        self.anchors = [(x, y) for y in ax for x in ax]  # row-major (y outer)
        if family == "parabola":
            mag = np.geomspace(0.004, 0.06, probe_shapes // 2)
            self.probe = np.concatenate([-mag[::-1], mag])
            dm = np.geomspace(0.004, 0.06, dense_shapes // 2)
            self.dense = np.concatenate([-dm[::-1], dm])
        else:
            self.probe = np.linspace(0.08 * size, 0.35 * size, probe_shapes)
            self.dense = np.linspace(0.08 * size, 0.35 * size, dense_shapes)
        self.register_buffer("bank1", build_bank(family, self.anchors, self.probe, size, size))
        self.encoder = SmallEncoder(ch)
        self.head1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1))
        self.head2 = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, 1, 1))
        self._bank2_cache = {}

    # stage-2 bank for one anchor, cached (anchors quantized to bins)
    def bank2_for(self, a_idx, device):
        if a_idx not in self._bank2_cache:
            a = self.anchors[a_idx]
            b = build_bank(self.family, [a], self.dense, self.size, self.size)[0]
            self._bank2_cache[a_idx] = b
        return self._bank2_cache[a_idx].to(device)

    def forward(self, img):
        B = img.shape[0]
        X = self.encoder(img)                                  # [B, C, H, W]
        feat = X.flatten(2)                                    # [B, C, HW]
        Y1 = vote(feat, self.bank1)                            # [B, C, A, S]
        Y1 = torch.logsumexp(Y1, dim=-1)                       # soft-max pool probes
        Y1 = Y1.reshape(B, -1, self.Ba, self.Ba)               # [B, C, Ba, Ba]
        P1 = self.head1(Y1)                                    # anchor logits
        # top-k anchors per image
        flat = P1.flatten(1)
        top = flat.topk(self.topk, dim=1).indices              # [B, k]
        P2, picked = [], []
        for b in range(B):
            row = []
            for a_idx in top[b].tolist():
                bank2 = self.bank2_for(a_idx, img.device)      # [S2, P]
                y2 = vote(feat[b:b + 1], bank2)                # [1, C, S2]
                row.append(self.head2(y2))                     # [1, 1, S2]
            P2.append(torch.cat(row, 0))                       # [k, 1, S2]
            picked.append(top[b])
        return P1, torch.stack(P2), torch.stack(picked), feat  # logits

    def anchor_index(self, x, y):
        """Nearest stage-1 anchor bin (row-major, y outer) for image coords."""
        scale = (self.Ba - 1) / (self.size - 1)
        xi = int(round(float(x) * scale))
        yi = int(round(float(y) * scale))
        xi = min(max(xi, 0), self.Ba - 1)
        yi = min(max(yi, 0), self.Ba - 1)
        return yi * self.Ba + xi

    def stage2_loss(self, feat, params, counts, sigma=1.0):
        """Teacher-forced BCE on the dense shape accumulator at GT anchors."""
        dense = torch.as_tensor(self.dense, dtype=torch.float32, device=feat.device)
        total, n = feat.new_zeros(()), 0
        for b in range(feat.shape[0]):
            for j in range(counts[b]):
                p = params[b][j]
                a_idx = self.anchor_index(p[0], p[1])
                bank2 = self.bank2_for(a_idx, feat.device)     # [S2, P]
                y2 = vote(feat[b:b + 1], bank2)                # [1, C, S2]
                logits = self.head2(y2)[0, 0]                  # [S2]
                s_bin = int((dense - float(p[2])).abs().argmin())
                idxs = torch.arange(len(dense), device=feat.device, dtype=torch.float32)
                tgt = torch.exp(-((idxs - s_bin) ** 2) / (2 * sigma ** 2))
                total = total + F.binary_cross_entropy_with_logits(logits, tgt)
                n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------- decoding
    @torch.no_grad()
    def detect(self, img, thresh=0.25):
        """thresh is a probability (sigmoid of the anchor logit)."""
        B = img.shape[0]
        X = self.encoder(img)
        feat = X.flatten(2)
        Y1 = vote(feat, self.bank1)
        Y1 = torch.logsumexp(Y1, dim=-1).reshape(B, -1, self.Ba, self.Ba)
        P1 = self.head1(Y1)                                    # [B, 1, Ba, Ba]
        # CenterNet-style NMS: keep local maxima of the 3x3 neighbourhood
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
                # soft-argmax over the 3x3 neighbourhood for sub-bin coords
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
                # stage 2 at the peak's bin anchor
                bank2 = self.bank2_for(a_idx, img.device)
                y2 = vote(feat[b:b + 1], bank2)
                s_prof = self.head2(y2)[0, 0]                  # [S2]
                s_idx = int(s_prof.argmax())
                lo, hi = max(0, s_idx - 1), min(len(self.dense), s_idx + 2)
                sw = torch.softmax(s_prof[lo:hi], 0)
                dvals = torch.as_tensor(self.dense[lo:hi], dtype=torch.float32,
                                        device=img.device)
                s_val = float((sw * dvals).sum())
                dets.append((ax, ay, s_val, v))
            out.append(dets)
        return out


# ------------------------------------------------------------------ losses
def gaussian_target(anchor_bins, gts, anchors_per_row, size, sigma=1.5):
    """2D anchor target maps from ground-truth anchor coords."""
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
        gaussian_target(Ba, [(p[0], p[1]) for p in params_batch[b][:counts[b]]], Ba, size)
        for b in range(B)]).to(P1.device)[:, None]
    return F.binary_cross_entropy_with_logits(
        P1, tgt, pos_weight=torch.tensor(8.0, device=P1.device))
