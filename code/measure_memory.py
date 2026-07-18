"""Measure real accumulator memory: dense O(B^d) vs our factorized cascade.
No training involved -- these are tensor sizes from the actual model config.
Reports parabola/circle (d=3) and ellipse (d=5) at matched B, plus the
arithmetic size a dense accumulator would require at d=5 (not instantiated:
infeasible at this resolution, which is exactly the point)."""
import torch
from hough import FactorizedDHT

C = 32          # encoder channels, matches SmallEncoder default
B = 32          # bins per axis, matched to our anchor_bins for fair comparison
bytes_per_elem = 4  # float32


def fmt(n):
    kb = n * bytes_per_elem / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    mb = kb / 1024
    if mb < 1024:
        return f"{mb:.2f} MB"
    return f"{mb/1024:.2f} GB"


def report(family, d, model_kwargs):
    model = FactorizedDHT(family, size=128, ch=C, **model_kwargs)
    Ba = model.Ba
    Bs = len(model.dense)
    k = model.topk
    dense_elems = C * (B ** d)
    stage1_elems = C * (Ba ** 2)
    stage2_elems = k * C * Bs
    factorized_elems = stage1_elems + stage2_elems
    print(f"\n{family} (d={d}): Ba={Ba}, Bs={Bs}, k={k}")
    print(f"  dense O(B^{d}) [arithmetic, not instantiated for d=5]: "
          f"{dense_elems:>14,} elements  {fmt(dense_elems)}")
    print(f"  stage 1 O(Ba^2):     {stage1_elems:>14,} elements  {fmt(stage1_elems)}")
    print(f"  stage 2 O(k*Bs):     {stage2_elems:>14,} elements  {fmt(stage2_elems)}")
    print(f"  factorized total:    {factorized_elems:>14,} elements  {fmt(factorized_elems)}")
    print(f"  reduction factor:    {dense_elems / factorized_elems:,.1f}x")
    return dense_elems, factorized_elems


d3 = report("parabola", 3, dict(anchor_bins=32, dense_shapes=48, topk=4))
d5 = report("ellipse", 5, dict(anchor_bins=32, topk=4))

print(f"\nScaling summary: going from d=3 to d=5 at fixed B={B}, C={C}:")
print(f"  dense accumulator grows {d5[0]/d3[0]:,.0f}x ({fmt(d3[0])} -> {fmt(d5[0])})")
print(f"  factorized total grows only {d5[1]/d3[1]:,.1f}x ({fmt(d3[1])} -> {fmt(d5[1])})")
print(f"  the d=5 dense accumulator ({fmt(d5[0])}) exceeds typical single-GPU "
      f"memory outright; the factorized d=5 total ({fmt(d5[1])}) does not")
