"""Shared train/val/test split for the real TuSimple experiments.

Used by train_real.py and the RANSAC baselines so every method is scored on
exactly the same test frames. The test frames are the first n//10 after a
seeded shuffle, the validation frames the next n//10, the rest train. For
seed 0 the test set is the same as in the earlier single-split code; the
validation set is new and comes out of the old training set.

Caveat: this is a random split over labeled frames. Frames from the same
drive can land on both sides, so real-data numbers can be a little optimistic.
"""
import numpy as np


def split_indices(n, seed=0):
    idx = list(range(n))
    np.random.default_rng(seed).shuffle(idx)
    k = max(1, n // 10)
    return idx[2 * k:], idx[k:2 * k], idx[:k]      # train, val, test
