"""Measure real accumulator memory: dense O(B^d) vs our factorized cascade.
No training involved -- these are tensor sizes from the actual model config."""
import torch
from hough import FactorizedDHT

C = 32          # encoder channels, matches SmallEncoder default
B = 32          # bins per axis, matched to our anchor_bins for fair comparison
d = 3           # parameter dimensionality (x0, y0, shape)

model = FactorizedDHT("parabola", size=128, ch=C, anchor_bins=32,
                      dense_shapes=48, topk=4)

Ba = model.Ba              # stage-1 anchor bins per axis
Bs = len(model.dense)       # stage-2 dense shape bins
k = model.topk

dense_elems = C * (B ** d)
stage1_elems = C * (Ba ** 2)
stage2_elems = k * C * Bs
factorized_elems = stage1_elems + stage2_elems

bytes_per_elem = 4  # float32

def fmt(n):
    kb = n * bytes_per_elem / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    return f"{kb/1024:.2f} MB"

print(f"config: C={C}, B={B} (dense per-axis bins), d={d}, "
      f"Ba={Ba}, Bs={Bs}, k={k}")
print(f"dense O(B^d):        {dense_elems:>10,} elements  {fmt(dense_elems)}")
print(f"stage 1 O(Ba^2):     {stage1_elems:>10,} elements  {fmt(stage1_elems)}")
print(f"stage 2 O(k*Bs):     {stage2_elems:>10,} elements  {fmt(stage2_elems)}")
print(f"factorized total:    {factorized_elems:>10,} elements  {fmt(factorized_elems)}")
print(f"reduction factor:    {dense_elems / factorized_elems:.1f}x")
