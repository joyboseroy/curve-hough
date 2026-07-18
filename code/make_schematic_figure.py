"""Schematic figure: dense O(B^d) accumulator vs. the factorized cascade.
Conceptual diagram, not data -- but numerically consistent with the
measured Table 1 numbers in the paper (26.9x, 152KB vs 4MB @ B=32, d=3).
Panel A is a hand-drawn isometric cube (not mplot3d) for reliable,
paper-quality rendering.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.linewidth": 0.8,
})

NAVY = "#1E2761"
ICE = "#CADCFC"
LIGHT = "#F4F7FE"
ACCENT = "#3D5AFE"
MUTED = "#5A6480"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.4))

# ---------- Panel A: dense accumulator, hand-drawn isometric cube ----------
axA.set_xlim(0, 10)
axA.set_ylim(0, 9)
axA.axis("off")
axA.set_aspect("equal")

ox, oy = 2.0, 2.2   # origin (front-bottom-left corner)
s = 4.6              # edge length (front face)
skew = np.array([0.55, 0.42]) * 3.2   # isometric offset for depth

front = np.array([[ox, oy], [ox+s, oy], [ox+s, oy+s], [ox, oy+s]])
top = np.array([[ox, oy+s], [ox+s, oy+s], [ox+s+skew[0], oy+s+skew[1]], [ox+skew[0], oy+s+skew[1]]])
side = np.array([[ox+s, oy], [ox+s+skew[0], oy+skew[1]], [ox+s+skew[0], oy+s+skew[1]], [ox+s, oy+s]])

axA.add_patch(mpatches.Polygon(front, closed=True, facecolor=ICE, edgecolor=NAVY, linewidth=1.2))
axA.add_patch(mpatches.Polygon(top, closed=True, facecolor="#E4ECFB", edgecolor=NAVY, linewidth=1.2))
axA.add_patch(mpatches.Polygon(side, closed=True, facecolor="#B4C9F2", edgecolor=NAVY, linewidth=1.2))

# grid lines on the front face to suggest a dense B x B grid
ngrid = 6
for i in range(1, ngrid):
    t = i / ngrid
    axA.plot([ox+t*s, ox+t*s], [oy, oy+s], color=NAVY, linewidth=0.35, alpha=0.5)
    axA.plot([ox, ox+s], [oy+t*s, oy+t*s], color=NAVY, linewidth=0.35, alpha=0.5)

axA.annotate("", xy=(ox+s+0.3, oy-0.15), xytext=(ox-0.3, oy-0.15),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axA.text(ox+s/2, oy-0.75, "anchor $x$", ha="center", fontsize=9.5, color=MUTED)
axA.annotate("", xy=(ox-0.75, oy+s+0.3), xytext=(ox-0.75, oy-0.3),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axA.text(ox-1.4, oy+s/2, "anchor $y$", ha="center", fontsize=9.5, color=MUTED, rotation=90)
axA.annotate("", xy=(ox+s+skew[0]+0.35, oy+skew[1]+0.35), xytext=(ox+s+0.05, oy+0.05),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axA.text(ox+s+skew[0]+0.55, oy+skew[1]-0.15, "shape", fontsize=9.5, color=MUTED)

axA.text(ox+s/2+skew[0]/2, oy+s+skew[1]+0.9,
         "Dense accumulator: $O(B^{d})$", ha="center", fontsize=11, fontweight="bold", color=NAVY)
axA.text(ox+s/2+skew[0]/2, oy-1.6,
         r"$32^3 = 1{,}048{,}576$ elements $\approx 4.00$ MB", ha="center", fontsize=9.5)

# ---------- Panel B: factorized cascade ----------
axB.set_xlim(0, 10)
axB.set_ylim(0, 9)
axB.axis("off")
axB.set_aspect("equal")

grid_n = 8
gx0, gy0, gs = 0.6, 4.6, 0.42
peak_ij = [(2, 5), (5, 2), (6, 6)]
for i in range(grid_n):
    for j in range(grid_n):
        c = ACCENT if (i, j) in peak_ij else LIGHT
        axB.add_patch(mpatches.Rectangle((gx0 + i*gs, gy0 + j*gs), gs, gs,
                                          facecolor=c, edgecolor=NAVY, linewidth=0.5))
axB.text(gx0 + grid_n*gs/2, gy0 - 0.55,
         "Stage 1: $O(B_a^{2})$ anchor map\n32,768 elements  128 KB",
         ha="center", fontsize=9.5)
axB.text(gx0 + grid_n*gs/2, gy0 + grid_n*gs + 0.3, "3 retained peaks",
         ha="center", fontsize=8.5, style="italic", color=MUTED)

peak_centers = [(gx0 + (i+0.5)*gs, gy0 + (j+0.5)*gs) for i, j in peak_ij]
bar_x0 = 7.0
bar_ys = [6.2, 4.2, 2.2]
for (px, py), by in zip(peak_centers, bar_ys):
    axB.annotate("", xy=(bar_x0 - 0.15, by + 0.35), xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                 connectionstyle="arc3,rad=0.15"))
    axB.add_patch(mpatches.Rectangle((bar_x0, by), 2.6, 0.7, facecolor=ICE,
                                      edgecolor=NAVY, linewidth=0.8))
    for k in range(11):
        axB.plot([bar_x0 + 0.08 + k*0.23]*2, [by+0.08, by+0.62], color=NAVY, linewidth=0.5, alpha=0.5)
axB.text(bar_x0 + 1.3, 1.5, "Stage 2: $O(kB_s)$ shape bins\n6,144 elements  24 KB",
          ha="center", fontsize=9.5)

axB.text(5.0, 8.5, "Factorized total: 38,912 elements $\\approx$ 152 KB",
          ha="center", fontsize=11, fontweight="bold", color=NAVY)
axB.text(5.0, 8.0, "(26.9$\\times$ smaller than dense)",
          ha="center", fontsize=9.5, color=MUTED)

plt.tight_layout()
plt.savefig("/home/claude/curve-hough/paper/figures/accumulator_schematic.pdf", bbox_inches="tight")
plt.savefig("/home/claude/curve-hough/paper/figures/accumulator_schematic.png", dpi=200, bbox_inches="tight")
print("saved")
