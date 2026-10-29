"""
plot_train_survival.py — Kaplan-Meier survival curve of a cohort, one point per biopsy.

No model is needed: this reads time and event from the labels CSV written by
define_labels.py, keeps the slides of one split (--split all for the full dataset),
aggregates them per biopsy, and saves:

  <output_dir>/km_curve_<split>.png         KM curve with 95% CI band, censoring
                                            ticks, n-year survival markers, a
                                            summary box and a number-at-risk table
  <output_dir>/survival_<split>.csv         one row per biopsy: patient, time, event
  <output_dir>/survival_summary_<split>.csv the statistics shown in the summary box

Statistics: number of slides, biopsies, patients, events and censored biopsies;
median follow-up (reverse Kaplan-Meier); median survival with 95% CI (or "not
reached"); and the survival probability, 95% CI, number at risk and cumulative
events at each --years horizon. A horizon beyond the longest follow-up is reported
as not estimable.

Usage (full dataset):
    python python_scripts/evaluate_MIL/plot_train_survival.py \\
        --labels_csv label_csvs/labels.csv --split all \\
        --output_dir results/cohort_survival --years 1 3 5 10
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

DAYS_PER_YEAR = 365.25
COLOR = "#2196F3"


def _first_time_at_or_below(t_km, curve, level=0.5):
    """First KM step time where ``curve`` drops to ``level`` or below, else NaN."""
    hit = np.flatnonzero(curve <= level)
    return float(t_km[hit[0]]) if len(hit) else float("nan")


def survival_summary(times, events, years):
    """Cohort statistics from per-sample times (in years) and event flags.

    Returns (summary, horizons): a dict of scalar statistics and a list of one dict
    per horizon in ``years``.
    """
    t_km, s_km, ci_lo, ci_hi = kaplan_meier(times, events)
    # Reverse KM (censoring as the event) gives the median *potential* follow-up,
    # which unlike the plain median of observed times is not shortened by events.
    t_rev, s_rev, _, _ = kaplan_meier(times, 1 - events)

    summary = {
        "n_events": int(events.sum()),
        "n_censored": int((events == 0).sum()),
        "median_follow_up": _first_time_at_or_below(t_rev, s_rev),
        "max_follow_up": float(times.max()),
        # Pointwise-band approach: the median's CI runs from where the lower band
        # crosses 0.5 to where the upper band does. NaN = not reached.
        "median_survival": _first_time_at_or_below(t_km, s_km),
        "median_survival_ci_lo": _first_time_at_or_below(t_km, ci_lo),
        "median_survival_ci_hi": _first_time_at_or_below(t_km, ci_hi),
    }

    horizons = []
    for y in years:
        row = {
            "years": y,
            "n_at_risk": int(np.sum(times >= y)),
            "events_before": int(np.sum((events == 1) & (times <= y))),
            "survival": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
        if y <= times.max():
            i = max(0, np.searchsorted(t_km, y, side="right") - 1)
            row.update(survival=s_km[i], ci_lo=ci_lo[i], ci_hi=ci_hi[i])
        horizons.append(row)
    return summary, horizons


def _fmt_years(v):
    return "not reached" if np.isnan(v) else f"{v:.1f} y"


def summary_lines(counts, summary, horizons):
    """Human-readable lines for the plot box and the console."""
    med = summary["median_survival"]
    med_ci = ""
    if not np.isnan(med):
        hi = summary["median_survival_ci_hi"]
        med_ci = (
            f" (95% CI {summary['median_survival_ci_lo']:.1f}–"
            f"{'NR' if np.isnan(hi) else f'{hi:.1f}'})"
        )
    lines = [
        f"Biopsies: {counts['n_biopsies']}   Patients: {counts['n_patients']}"
        f"   Slides: {counts['n_slides']}",
        f"Events: {summary['n_events']}   Censored: {summary['n_censored']}",
        f"Median follow-up: {_fmt_years(summary['median_follow_up'])}"
        f"  (max {summary['max_follow_up']:.1f} y)",
        f"Median survival: {_fmt_years(med)}{med_ci}",
    ]
    for h in horizons:
        if np.isnan(h["survival"]):
            lines.append(f"{h['years']:g}-year survival: not estimable")
        else:
            lines.append(
                f"{h['years']:g}-year survival: {100 * h['survival']:.1f}% "
                f"({100 * h['ci_lo']:.1f}–{100 * h['ci_hi']:.1f})"
                f"   at risk {h['n_at_risk']}"
            )
    return lines


def plot_km(times, events, horizons, lines, title, path):
    """KM curve (time in years) with CI band, horizon markers, summary box and
    number-at-risk table."""
    t_km, s_km, ci_lo, ci_hi = kaplan_meier(times, events)
    t_max = float(times.max())

    fig, (ax, ax_t) = plt.subplots(
        2, 1, figsize=(9, 6.2), gridspec_kw={"height_ratios": [5, 0.6]}, sharex=True
    )
    ax.step(t_km, s_km, where="post", color=COLOR, linewidth=2, label="Kaplan-Meier")
    ax.fill_between(
        np.append(t_km, t_max),
        np.append(ci_lo, ci_lo[-1]),
        np.append(ci_hi, ci_hi[-1]),
        step="post",
        color=COLOR,
        alpha=0.15,
        label="95% CI (Greenwood)",
    )
    censor_times = times[events == 0]
    censor_s = s_km[np.maximum(0, np.searchsorted(t_km, censor_times, side="right") - 1)]
    ax.scatter(
        censor_times, censor_s, marker="|", color=COLOR, s=40, linewidths=1.2,
        zorder=3, label="Censored",
    )

    # n-year survival markers
    for h in horizons:
        if np.isnan(h["survival"]):
            continue
        ax.plot([h["years"]] * 2, [0, h["survival"]], ls=":", color="gray", lw=1)
        ax.plot(h["years"], h["survival"], "o", color=COLOR, ms=5, zorder=4)
        ax.annotate(
            f"{100 * h['survival']:.0f}%",
            (h["years"], h["survival"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color="#333333",
        )

    # Survival curves of these cohorts stay high, so the empty lower right is the
    # spot that does not cover the curve.
    ax.text(
        0.98, 0.03, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
        multialignment="left", fontsize=8, family="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc="#C62828", ec="#8E1B1B", alpha=0.9),
    )
    ax.set_ylabel("Event-free survival probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, t_max * 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(title)

    # Number-at-risk table at whole-year ticks
    step = max(1, int(np.ceil(t_max / 10)))
    ticks = np.arange(0, t_max + 1e-9, step)
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([])
    for side in ("top", "right", "left"):
        ax_t.spines[side].set_visible(False)
    for tt in ticks:
        ax_t.text(tt, 0.5, str(int(np.sum(times >= tt))), ha="center", va="center",
                  fontsize=8, color=COLOR)
    ax_t.set_ylabel("At risk", rotation=0, ha="right", va="center", fontsize=8,
                    color="gray", labelpad=16)
    ax_t.set_xticks(ticks)
    ax_t.set_xlabel("Time since biopsy (years)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KM curve saved to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Kaplan-Meier survival curve of a cohort, one point per biopsy."
    )
    p.add_argument(
        "--labels_csv", required=True, help="Labels CSV written by define_labels.py."
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which slides of the labels CSV to plot ('all' = full dataset).",
    )
    p.add_argument("--output_dir", default="results/train_survival")
    p.add_argument(
        "--years",
        nargs="+",
        type=float,
        default=[1, 3, 5, 10],
        help="Horizons (years) at which to report survival.",
    )
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
    if table.empty:
        raise RuntimeError(f"No slides in the '{args.split}' split of {args.labels_csv}.")
    names = list(table.index)
    times = table["time"].to_numpy(dtype=float) / DAYS_PER_YEAR
    events = table["event"].to_numpy(dtype=int)

    counts = {
        "n_slides": len(names),
        "n_biopsies": table["biopsy_number"].nunique(),
        "n_patients": table["patient"].nunique() if "patient" in table else "n/a",
    }
    if args.no_aggregate:
        ids, id_col = names, "bag_name"
        patients = table["patient"].to_numpy() if "patient" in table else None
    else:
        # Risks don't exist here, so we pass zeros and discard the aggregated risk.
        _, times, events, ids = aggregate_by_biopsy(
            np.zeros_like(times), times, events, names, table
        )
        id_col = "biopsy"
        patients = (
            table.groupby("biopsy_number")["patient"].first().loc[ids].to_numpy()
            if "patient" in table
            else None
        )

    summary, horizons = survival_summary(times, events, args.years)
    lines = summary_lines(counts, summary, horizons)
    print("\n".join(lines))

    data = pd.DataFrame({id_col: ids, "time_years": times, "event": events})
    if patients is not None:
        data.insert(1, "patient", patients)
    data_path = os.path.join(args.output_dir, f"survival_{args.split}.csv")
    data.to_csv(data_path, index=False)

    rows = [{"statistic": k, "value": v} for k, v in {**counts, **summary}.items()]
    for h in horizons:
        y = f"{h['years']:g}y"
        rows += [
            {"statistic": f"survival_{y}", "value": h["survival"]},
            {"statistic": f"survival_{y}_ci_lo", "value": h["ci_lo"]},
            {"statistic": f"survival_{y}_ci_hi", "value": h["ci_hi"]},
            {"statistic": f"n_at_risk_{y}", "value": h["n_at_risk"]},
            {"statistic": f"events_before_{y}", "value": h["events_before"]},
        ]
    summary_path = os.path.join(args.output_dir, f"survival_summary_{args.split}.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Per-{id_col} table → {data_path}\nSummary → {summary_path}")

    split_name = {
        "all": "Full dataset", "train": "Training set", "val": "Validation set",
        "test": "Test set",
    }
    plot_km(
        times, events, horizons, lines,
        f"{split_name[args.split]} — Kaplan-Meier survival "
        f"(one point per {'slide' if args.no_aggregate else 'biopsy'})",
        os.path.join(args.output_dir, f"km_curve_{args.split}.png"),
    )


if __name__ == "__main__":
    main()
