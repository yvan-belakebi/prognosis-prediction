"""
plot_train_survival.py — Plot the Kaplan-Meier survival curve of the training cohort.

No model is needed: this just reads the survival labels (.npy files) under
--labels_paths, optionally filters them to the slides listed in --train_csv,
aggregates per biopsy, and saves a KM plot with a 95% CI band.

Usage (filtered to a specific cohort):
    python python_scripts/MIL/plot_train_survival.py \\
        --labels_paths WSI/IgA/labels WSI/IgA_registry/labels \\
        --train_csv followup_data/survival_training_files.csv \\
        --output_dir results/train_survival

Usage (all labels under the given paths):
    python python_scripts/MIL/plot_train_survival.py \\
        --labels_paths WSI/IgA/labels WSI/IgA_registry/labels \\
        --output_dir results/all_survival
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the KM estimator and biopsy aggregation from evaluate_survival.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from evaluate_survival import kaplan_meier, aggregate_by_biopsy


def load_train_names(csv_path):
    """Read training slide basenames from a CSV (first column or 'bag_name')."""
    df = pd.read_csv(csv_path)
    col = "bag_name" if "bag_name" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.replace(".npy", "", regex=False))


def collect_labels(labels_paths, train_names=None):
    """Load (name, time, event) for every .npy label found under labels_paths.

    If train_names is None, all .npy labels are loaded. Otherwise only labels
    whose relative path (or filename stem) appears in train_names are kept.
    """
    names, times, events = [], [], []
    for lp in labels_paths:
        for root, _, files in os.walk(lp):
            for f in files:
                if not f.endswith(".npy"):
                    continue
                stem = os.path.splitext(f)[0]
                rel = os.path.relpath(os.path.join(root, stem), lp).replace(os.sep, "/")
                if train_names is not None and rel not in train_names and stem not in train_names:
                    continue
                y = np.load(os.path.join(root, f))
                if y.ndim == 2:  # per-patch label array — labels are constant per bag
                    y = y[0]
                names.append(rel)
                times.append(float(y[0]))
                events.append(int(y[1]))
    return np.array(names), np.array(times), np.array(events)


def plot_km(times, events, output_dir, time_unit="days"):
    """KM curve with Greenwood 95% CI and censoring ticks."""
    t_km, s_km, ci_lo, ci_hi = kaplan_meier(times, events)
    t_max = float(times.max()) if len(times) else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    color = "#2196F3"

    ax.step(
        t_km,
        s_km,
        where="post",
        color=color,
        linewidth=2,
        label=f"Training cohort  (n={len(times)}, events={int(events.sum())})",
    )

    # Extend the last CI segment to the right edge for aesthetics
    t_fill = np.append(t_km, t_max)
    lo_fill = np.append(ci_lo, ci_lo[-1])
    hi_fill = np.append(ci_hi, ci_hi[-1])
    ax.fill_between(
        t_fill,
        lo_fill,
        hi_fill,
        step="post",
        color=color,
        alpha=0.15,
        label="95% CI (Greenwood)",
    )

    # Censoring ticks at the KM step value for each censored time
    censor_times = times[events == 0]
    censor_s = [
        s_km[max(0, np.searchsorted(t_km, tc, side="right") - 1)] for tc in censor_times
    ]
    ax.scatter(
        censor_times,
        censor_s,
        marker="|",
        color=color,
        s=40,
        linewidths=1.2,
        zorder=3,
        label="Censored",
    )

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("Training cohort — Kaplan-Meier survival")

    fig.tight_layout()
    path = os.path.join(output_dir, "train_km_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KM curve saved to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Plot the Kaplan-Meier survival curve of the training cohort."
    )
    p.add_argument("--labels_paths", nargs="+", required=True)
    p.add_argument(
        "--train_csv",
        default=None,
        help="Optional CSV listing slide basenames to keep. "
        "If omitted, every .npy label found under --labels_paths is used.",
    )
    p.add_argument("--output_dir", default="results/train_survival")
    p.add_argument("--time_unit", default="days")
    p.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Skip biopsy-level aggregation; treat every slide as its own sample.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_csv is not None:
        train_names = load_train_names(args.train_csv)
        print(f"Filtering to {len(train_names)} names from {args.train_csv}")
    else:
        train_names = None
        print("No --train_csv given; loading every label under --labels_paths")

    names, times, events = collect_labels(args.labels_paths, train_names)
    if len(names) == 0:
        raise RuntimeError(
            "No matching labels found — check --labels_paths"
            + (" and --train_csv." if args.train_csv else ".")
        )
    print(f"Slides loaded: {len(names)}  |  events: {int(events.sum())}")

    # Aggregate slides to biopsies (mirrors evaluate_survival.py). Risks don't
    # exist here, so we pass zeros and discard the aggregated risk output.
    if not args.no_aggregate and any("/" in n for n in names):
        _, times, events, biopsies = aggregate_by_biopsy(
            np.zeros_like(times), times, events, list(names)
        )
        print(
            f"Aggregated to {len(biopsies)} biopsies  |  events: {int(events.sum())}"
        )
        out_ids = biopsies
        id_col = "biopsy"
    else:
        out_ids = list(names)
        id_col = "bag_name"

    # Save the underlying data alongside the figure
    csv_path = os.path.join(args.output_dir, "train_survival.csv")
    pd.DataFrame({id_col: out_ids, "time": times, "event": events}).to_csv(
        csv_path, index=False
    )
    print(f"Survival table → {csv_path}")

    plot_km(times, events, args.output_dir, time_unit=args.time_unit)


if __name__ == "__main__":
    main()