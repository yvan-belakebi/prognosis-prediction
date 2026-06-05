"""
compare_regression_losses.py — Visual comparison of regression MIL training curves.

Reads loss_log.csv from each model's log directory and overlays train loss,
val loss, val MAE, and val RMSE in side-by-side subplots.

Usage:
    python python_scripts/MIL/compare_regression_losses.py
    python python_scripts/MIL/compare_regression_losses.py \\
        --abmil_dir    results/losses_regression_abmil \\
        --dsmil_dir    results/losses_regression_dsmil \\
        --transmil_dir results/losses_regression_transmil \\
        --output       regression_loss_comparison.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults — match the directory names the user described
# ---------------------------------------------------------------------------
DEFAULT_DIRS = {
    "ABMIL": "results/losses_regression_abmil",
    "DSMIL": "results/losses_regression_dsmil",
    "TransMIL": "results/losses_regression_transmil",
}
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]

METRICS = [
    ("train_loss", "Train loss"),
    ("val_loss", "Val loss"),
    ("val_mae", "Val MAE"),
    ("val_rmse", "Val RMSE"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_log(log_dir):
    path = os.path.join(log_dir, "loss_log.csv")
    df = pd.read_csv(path)
    for col in ("val_loss", "val_mae", "val_rmse"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pretrain_span(df):
    """Return (epoch_start, epoch_end) of the pretrain phase, or None."""
    pre = df[df["phase"] == "pretrain"]["epoch"]
    return (pre.iloc[0], pre.iloc[-1]) if not pre.empty else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare regression MIL loss curves across model architectures."
    )
    parser.add_argument("--abmil_dir", default=DEFAULT_DIRS["ABMIL"])
    parser.add_argument("--dsmil_dir", default=DEFAULT_DIRS["DSMIL"])
    parser.add_argument("--transmil_dir", default=DEFAULT_DIRS["TransMIL"])
    parser.add_argument(
        "--output",
        default="regression_loss_comparison.png",
        help="Output image path (default: regression_loss_comparison.png).",
    )
    args = parser.parse_args()

    model_dirs = {
        "ABMIL": args.abmil_dir,
        "DSMIL": args.dsmil_dir,
        "TransMIL": args.transmil_dir,
    }

    # Load, skip models whose log is missing
    data = {}
    for name, d in model_dirs.items():
        try:
            data[name] = load_log(d)
            print(f"Loaded {name}: {d}/loss_log.csv ({len(data[name])} rows)")
        except FileNotFoundError:
            print(f"Warning: {d}/loss_log.csv not found — skipping {name}.")

    if not data:
        raise RuntimeError("No loss_log.csv files found. Check the directory paths.")

    # Only draw subplots for metrics that have at least one non-NaN value
    active = [
        (col, label)
        for col, label in METRICS
        if any(col in df.columns and df[col].notna().any() for df in data.values())
    ]

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, active):
        for (name, df), color in zip(data.items(), COLORS):
            if col not in df.columns:
                continue
            valid = df[col].notna()
            ax.plot(
                df.loc[valid, "epoch"],
                df.loc[valid, col],
                label=name,
                color=color,
                linewidth=1.8,
            )

        # Shade pretrain region for each model that has one (use first model's span
        # per subplot to avoid overlapping identical shades).
        for name, df in data.items():
            span = pretrain_span(df)
            if span is not None:
                ax.axvspan(
                    span[0],
                    span[1],
                    alpha=0.07,
                    color="gray",
                    zorder=0,
                )
            break  # one shade is enough — all models share the same x-axis

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    # Add a note about the pretrain shade if any model used it
    if any(pretrain_span(df) is not None for df in data.values()):
        fig.text(
            0.5,
            -0.01,
            "Gray band = pretrain phase",
            ha="center",
            fontsize=8,
            color="gray",
        )

    fig.suptitle("Regression MIL — loss comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
