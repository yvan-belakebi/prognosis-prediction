"""
bag_size_stats.py — Patch-count distribution of a feature directory, and what it
costs in GPU memory once bags are padded into a batch.

Padded-batch models (DeepGraphSurv, PatchGCN) size every activation by the
LARGEST slide in the batch, so a run OOMs "occasionally": only the batches that
happen to catch a big slide are expensive. This reads h5/npy headers (no patch
data), reports the distribution, and simulates how large the per-batch maximum
gets — which is what --max_patches has to cap.

Usage:
    python python_scripts/explore_data/bag_size_stats.py \
        WSI/non_IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested \
        --batch_size 12 --budget_mb 21766
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MIL"))
from mil_utils import discover_bags, _bag_shape  # noqa: E402

MB = 1024 ** 2
# ChebConv keeps K copies of the (B, N, C) input plus the stacked tensor, for the
# representation and attention branch, and autograd holds them all until backward.
CHEB_COPIES = 22


def batch_peak_mb(b, n, feat_dim, k=5):
    """Rough peak activation + dense-adjacency cost of one DeepGraphSurv step."""
    activations = CHEB_COPIES * b * n * feat_dim * 4
    dense_adj = b * n * n * 4
    return (activations + dense_adj) / MB


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dirs", nargs="+", help="Feature directories to scan.")
    p.add_argument("--file_ext", default=".h5")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--k", type=int, default=5, help="Chebyshev order (--K in MIL.py).")
    p.add_argument("--budget_mb", type=int, default=None, help="Allocatable GPU memory (nvidia-smi memory.free at idle).")
    p.add_argument("--trials", type=int, default=2000, help="Random batches to simulate.")
    args = p.parse_args()

    sizes, feat_dim = [], None
    for d in args.features_dirs:
        for name in discover_bags(d, extensions=(args.file_ext,)):
            shape = _bag_shape(os.path.join(d, name + args.file_ext))
            if len(shape) >= 2 and shape[0] > 0:
                sizes.append(shape[0])
                feat_dim = feat_dim or shape[1]
    if not sizes:
        sys.exit("No non-empty bags found.")
    sizes = np.array(sizes)

    print(f"{len(sizes)} bags | feat_dim {feat_dim}")
    qs = [50, 75, 90, 95, 99, 99.9, 100]
    print("patches per bag: " + "  ".join(f"p{q:g}={np.percentile(sizes, q):.0f}" for q in qs))

    rng = np.random.default_rng(0)
    batch_max = sizes[rng.integers(0, len(sizes), size=(args.trials, args.batch_size))].max(axis=1)
    print(
        f"\nmax bag per batch of {args.batch_size}: "
        + "  ".join(f"p{q:g}={np.percentile(batch_max, q):.0f}" for q in [50, 90, 99, 100])
    )

    print(f"\nEstimated peak per step (batch {args.batch_size}, K={args.k}, feat_dim {feat_dim}):")
    for cap in [None, 8192, 4096, 2048, 1024]:
        n = int(batch_max.max()) if cap is None else min(cap, int(batch_max.max()))
        peak = batch_peak_mb(args.batch_size, n, feat_dim, args.k)
        label = "no cap" if cap is None else f"--max_patches {cap}"
        verdict = ""
        if args.budget_mb:
            verdict = "  OK" if peak < args.budget_mb * 0.85 else "  OVER BUDGET"
        print(f"  {label:>20}: N={n:>6}  ~{peak:>8.0f} MB{verdict}")
        if cap is not None:
            affected = (sizes > cap).mean() * 100
            print(f"  {'':>20}  subsamples {affected:.1f}% of bags")


if __name__ == "__main__":
    main()
