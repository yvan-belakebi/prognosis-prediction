"""
plot_train_survival.py — Plot the Kaplan-Meier survival curve of the training cohort.

No model is needed: this reads time and event from the labels CSV written by
define_labels.py, keeps the slides of one split, aggregates per biopsy, and saves a
KM plot with a 95% CI band.

Usage:
    python python_scripts/evaluate_MIL/plot_train_survival.py \\
        --labels_csv label_csvs/labels.csv \\
        --output_dir results/train_survival
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
from mil_utils import load_labels_table


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
    p.add_argument(
        "--labels_csv", required=True, help="Labels CSV written by define_labels.py."
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "all"],
        help="Which slides of the labels CSV to plot.",
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

    table = load_labels_table(args.labels_csv)
    if args.split != "all":
        table = table[table["split"] == args.split]
    names = list(table.index)
    times = table["time"].to_numpy(dtype=float)
    events = table["event"].to_numpy(dtype=int)
    if len(names) == 0:
        raise RuntimeError(f"No slides in the '{args.split}' split of {args.labels_csv}.")
    print(f"Slides loaded: {len(names)}  |  events: {int(events.sum())}")

    # Aggregate slides to biopsies (mirrors evaluate_survival.py). Risks don't
    # exist here, so we pass zeros and discard the aggregated risk output.
    if not args.no_aggregate:
        _, times, events, biopsies = aggregate_by_biopsy(
            np.zeros_like(times), times, events, names, table
        )
        print(
            f"Aggregated to {len(biopsies)} biopsies  |  events: {int(events.sum())}"
        )
        out_ids = biopsies
        id_col = "biopsy"
    else:
        out_ids = names
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