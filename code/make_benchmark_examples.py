"""Benchmark examples figure: real generated images (no trained model
needed), clean vs. corrupted, across all three families. Pure dataset
generator output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset import SyntheticCurves

families = ["parabola", "circle", "ellipse"]
fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.2))

for col, family in enumerate(families):
    clean = SyntheticCurves(family, size=64, n_images=4, seed=7,
                            noise=0.0, occlude=False, distractors=False, max_curves=1)
    hard = SyntheticCurves(family, size=64, n_images=4, seed=7)  # default corrupted

    img_clean, _, _ = clean[0]
    img_hard, _, _ = hard[1]  # a different draw for visual variety

    axes[0, col].imshow(img_clean[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{family}\n(clean)", fontsize=11)
    axes[0, col].axis("off")

    axes[1, col].imshow(img_hard[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title("(corrupted: noise,\nocclusion, distractors)", fontsize=9.5)
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("../paper/figures/benchmark_examples.pdf", bbox_inches="tight")
plt.savefig("../paper/figures/benchmark_examples.png", dpi=200, bbox_inches="tight")
print("saved")
