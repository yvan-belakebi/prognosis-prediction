"""
list_large_slides.py — Report slides whose patch count exceeds a threshold.

Scans a coords directory (one .npy file per slide containing patch coordinates)
and lists slides that would produce adjacency matrices larger than a given
memory budget.

Usage:
    python python_scripts/list_large_slides.py \
        --coords_dir WSI/IgA/coords \
        --max_patches 4096 \
        --budget_gb 4
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--coords_dir", required=True,
                        help="Directory of per-slide coordinate .npy files.")
    parser.add_argument("--max_patches", type=int, default=4096,
                        help="Subsampling threshold used during training / eval.")
    parser.add_argument("--budget_gb", type=float, default=4.0,
                        help="Flag slides whose dense adj would exceed this many GB.")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.coords_dir) if f.endswith(".npy"))
    print(f"Scanning {len(files)} slides in {args.coords_dir} ...\n")

    rows = []
    for fname in files:
        coords = np.load(os.path.join(args.coords_dir, fname))
        n = len(coords)
        adj_gb = n * n * 4 / 1024**3
        rows.append((fname, n, adj_gb))

    rows.sort(key=lambda x: -x[1])

    flagged = [(f, n, g) for f, n, g in rows if g > args.budget_gb]
    print(f"{'Slide':<50}  {'Patches':>8}  {'Adj (GB)':>9}  {'> budget?':>10}")
    print("-" * 82)
    for fname, n, adj_gb in rows[:30]:
        flag = "  *** TOO LARGE" if adj_gb > args.budget_gb else ""
        print(f"{fname:<50}  {n:>8d}  {adj_gb:>9.2f}{flag}")

    if len(rows) > 30:
        print(f"  ... ({len(rows) - 30} more slides not shown)")

    print(f"\nSlides exceeding {args.budget_gb} GB budget: {len(flagged)}")
    print(f"Slides exceeding --max_patches {args.max_patches}: "
          f"{sum(1 for _, n, _ in rows if n > args.max_patches)}")
    if flagged:
        print(f"\nLargest slide: {flagged[0][0]}  ({flagged[0][1]:,} patches, "
              f"{flagged[0][2]:.1f} GB dense adj)")
        print(f"With --max_patches {args.max_patches} that becomes "
              f"{args.max_patches**2 * 4 / 1024**2:.0f} MB.")


if __name__ == "__main__":
    main()
