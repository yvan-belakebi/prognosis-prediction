"""plot_vote_vs_stains.py — Fusion vs each single-stain model, on the test set.

Reads the paired comparisons vote_across_stains.sh writes with compare_cindex.py,
results/compare/<fusion>_vs_<stain>_test.txt. Each one is computed on the test
biopsies that both models scored, i.e. the biopsies that have that stain, so every
row compares the stain with the fusion on that stain's biopsies (N differs per row).

One row per stain: the stain's C-index (grey) and the fusion's (colour) on the same
biopsies, with the difference, its bootstrap 95% CI and p-value on the right.

Usage:
    python python_scripts/evaluate_MIL/plot_vote_vs_stains.py                  # voting_none
    python python_scripts/evaluate_MIL/plot_vote_vs_stains.py --fusion late_fusion
    python python_scripts/evaluate_MIL/plot_vote_vs_stains.py --fusion voting_none_c1q \\
        --stains HE_HES_c1q PAS_c1q PASM_c1q
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse(path):
    """N, events, C-index A and B, bootstrap CI and p of the difference, from one file."""
    text = open(path, encoding="utf-8").read()
    num = r"([+-]?\d+\.\d+)"
    n, events = map(int, re.search(r"N=(\d+), events=(\d+)", text).groups())
    boot = text.split("[Bootstrap]", 1)[1]
    lo, hi = map(float, re.search(rf"95% CI for \(A − B\): \[{num}, {num}\]", boot).groups())
    return dict(n=n, events=events,
                c_fusion=float(re.search(rf"C-index A: {num}", text).group(1)),
                c_stain=float(re.search(rf"C-index B: {num}", text).group(1)),
                ci_low=lo, ci_high=hi,
                p=float(re.search(rf"p-value: {num}", boot).group(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare_dir", default="results/compare")
    ap.add_argument("--fusion", default="voting_none",
                    help="Fusion name as in the file names: voting_none, voting_cindex, "
                    "late_fusion, with _shared and/or the TAG_SUFFIX appended.")
    ap.add_argument("--stains", nargs="+", default=None,
                    help="Stain names as in the file names (default: every file found). "
                    "Older runs may have left files for stains no longer in STAINS.")
    ap.add_argument("--output", default=None,
                    help="Figure path without extension (default: <compare_dir>/<fusion>_vs_stains).")
    args = ap.parse_args()

    prefix = f"{args.fusion}_vs_"
    if args.stains:
        paths = {s: os.path.join(args.compare_dir, f"{prefix}{s}_test.txt") for s in args.stains}
    else:
        paths = {os.path.basename(p)[len(prefix):-len("_test.txt")]: p
                 for p in glob.glob(os.path.join(args.compare_dir, f"{prefix}*_test.txt"))}
        # Fusion-vs-fusion files (late_fusion_vs_voting_*) are not single stains.
        paths = {s: p for s, p in paths.items() if not s.startswith(("voting_", "late_fusion"))}
    rows = []
    for stain, path in paths.items():
        if not os.path.isfile(path):
            print(f"--- {stain}: no {path}, skipped")
            continue
        rows.append(dict(stain=stain, **parse(path)))
    if not rows:
        raise SystemExit(f"No {prefix}*_test.txt in {args.compare_dir}")
    df = pd.DataFrame(rows).sort_values("c_stain").reset_index(drop=True)
    print(df.to_string(index=False, float_format="%.4f"))

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.6))
    y = range(len(df))
    ax.hlines(y, df.c_stain, df.c_fusion, color="#C9CFD7", lw=3, zorder=1)
    ax.scatter(df.c_stain, y, s=70, color="#8A93A0", zorder=2, label="single stain")
    ax.scatter(df.c_fusion, y, s=70, color="#E0609A", zorder=3, label=args.fusion)
    ax.set_yticks(list(y), [f"{s}  (n={n}, {e} ev.)" for s, n, e in zip(df.stain, df.n, df.events)])
    for i, r in df.iterrows():
        star = "*" if r.p < 0.05 else ""
        ax.text(1.02, i, f"{r.c_fusion - r.c_stain:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                f"  p={r.p:.3f}{star}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_xlabel("Test C-index, on the biopsies with the stain")
    ax.set_title(f"{args.fusion} vs each single stain (difference, bootstrap 95% CI)", fontsize=11)
    ax.grid(axis="x", color="#E3E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    out = args.output or os.path.join(args.compare_dir, f"{args.fusion}_vs_stains")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.png and .pdf")


if __name__ == "__main__":
    main()
