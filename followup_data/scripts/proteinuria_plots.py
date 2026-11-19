"""
proteinuria_plots.py — Describe proteinuria values and compare them across sampling methods.

Input: the CSV written by proteinuria.py (columns Proteinuria_g_d, Sampling_method).
Output, in --out_dir:
    proteinuria_overall.png        histogram of all values (g/24 h), with median
    proteinuria_methods_count.png  number of patients per sampling method (incl. none)
    proteinuria_by_method.png      box plot + points of the values, one box per method
    proteinuria_ecdf.png           cumulative distribution, one line per method
    proteinuria_summary.csv        n, mean, sd, median, IQR, min, max per method
                                   (+ Kruskal-Wallis test across methods, printed)

Note: dipstick values are discrete (0, 1.105, 3.4, 5.95 g/day), so their box and
ECDF steps reflect the conversion table rather than a measured distribution.

Example:
    python followup_data/scripts/proteinuria_plots.py proteinuria.csv --out_dir plots/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

VALUE, METHOD = "Proteinuria_g_d", "Sampling_method"
# preference order of proteinuria.py; colour follows the method, never its rank
METHODS = {
    "24h_urine_g_d":  ("24-h urine (g/day)", "#2a78d6"),
    "urine_conc_g_l": ("Urine conc. (g/L)", "#eb6834"),
    "PCR":            ("PCR (Weaver → ACR)", "#1baf7a"),
    "ACR":            ("ACR", "#eda100"),
    "dipstick":       ("Dipstick", "#e87ba4"),
}
NONE_LABEL, NONE_COLOR = "No measure", "#a3a29b"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
UNIT = "Proteinuria (g/24 h)"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def present_methods(df):
    return [m for m in METHODS if (df[METHOD] == m).any()]


def summary(df):
    g = df.dropna(subset=[VALUE]).groupby(METHOD)[VALUE]
    s = g.agg(n="count", mean="mean", sd="std", median="median",
              q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), min="min", max="max")
    s = s.reindex(present_methods(df))
    total = df[VALUE].dropna()
    s.loc["all"] = [total.count(), total.mean(), total.std(), total.median(),
                    total.quantile(.25), total.quantile(.75), total.min(), total.max()]
    s["n"] = s["n"].astype(int)
    return s.round(3)


def plot_overall(df, ax):
    v = df[VALUE].dropna()
    ax.hist(v, bins=np.arange(0, max(v.max(), 1) + 0.5, 0.5), color=METHODS["24h_urine_g_d"][1],
            edgecolor="white", linewidth=1)
    med = v.median()
    ax.axvline(med, color=INK, linestyle="--", linewidth=1.2)
    ax.annotate(f"median {med:.2f}", (med, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points", va="top", color=INK)
    ax.set(xlabel=UNIT, ylabel="Patients", title=f"Proteinuria at first biopsy (n = {len(v)})")
    ax.grid(axis="x", visible=False)


def plot_counts(df, ax):
    methods = present_methods(df)
    counts = [(df[METHOD] == m).sum() for m in methods] + [df[METHOD].isna().sum()]
    labels = [METHODS[m][0] for m in methods] + [NONE_LABEL]
    colors = [METHODS[m][1] for m in methods] + [NONE_COLOR]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, counts, color=colors, height=0.7)
    total = sum(counts)
    for yi, c in zip(y, counts):
        ax.annotate(f"{c} ({100 * c / total:.1f}%)", (c, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Patients", title="Sampling method used for proteinuria")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)


def plot_by_method(df, ax, rng):
    methods = present_methods(df)
    data = [df.loc[df[METHOD] == m, VALUE].dropna() for m in methods]
    bp = ax.boxplot(data, positions=range(len(methods)), widths=0.5, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, linewidth=1.5))
    for patch, m in zip(bp["boxes"], methods):
        patch.set(facecolor=METHODS[m][1], alpha=0.35, edgecolor=METHODS[m][1])
    for i, (m, d) in enumerate(zip(methods, data)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=METHODS[m][1],
                   alpha=0.35, linewidths=0)
    short = lambda m: METHODS[m][0].replace(" (", "\n(")  # wrap units onto a 2nd line
    ax.set_xticks(range(len(methods)), [f"{short(m)}\nn = {len(d)}" for m, d in zip(methods, data)])
    ax.set(ylabel=UNIT, title="Proteinuria by sampling method")
    ax.grid(axis="x", visible=False)


def plot_ecdf(df, ax):
    for m in present_methods(df):
        v = np.sort(df.loc[df[METHOD] == m, VALUE].dropna())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", linewidth=2,
                color=METHODS[m][1], label=f"{METHODS[m][0]} (n = {len(v)})")
    ax.set(xlabel=UNIT, ylabel="Cumulative proportion of patients", ylim=(0, 1.02),
           title="Cumulative distribution by sampling method")
    ax.legend(frameon=False, loc="lower right")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="output of proteinuria.py")
    ap.add_argument("--out_dir", default=".", help="where plots and summary are written")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    make_all(df, out)


def make_all(df, out):
    rng = np.random.default_rng(0)  # reproducible jitter
    for name, fn, size in [("overall", plot_overall, (6, 4)),
                           ("methods_count", plot_counts, (6, 3.5)),
                           ("by_method", lambda d, a: plot_by_method(d, a, rng), (7, 4.5)),
                           ("ecdf", plot_ecdf, (6.5, 4.5))]:
        fig, ax = plt.subplots(figsize=size)
        fn(df, ax)
        fig.savefig(out / f"proteinuria_{name}.png")
        plt.close(fig)

    s = summary(df)
    s.to_csv(out / "proteinuria_summary.csv")
    print(s.to_string())
    groups = [df.loc[df[METHOD] == m, VALUE].dropna() for m in present_methods(df)]
    groups = [g for g in groups if len(g)]
    if len(groups) > 1:
        h, p = kruskal(*groups)
        print(f"\nKruskal-Wallis across methods: H = {h:.2f}, p = {p:.3g}")


if __name__ == "__main__":
    main()
