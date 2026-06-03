"""
evaluate_survival.py — Evaluate a trained MIL survival model on the validation set.

Produces:
  <output_dir>/km_curves.png        Kaplan-Meier curves stratified by predicted risk group
  <output_dir>/risk_distribution.png Histogram of risk scores split by event status
  <output_dir>/risk_scores.csv       Per-slide risk score, time, and event

C-index and log-rank p-value are printed to stdout.

Usage:
    python python_scripts/MIL/evaluate_survival.py \\
        --model_type abmil \\
        --checkpoint checkpoints/abmil_model.pth \\
        --features_paths WSI/IgA/UNI2-h_feats WSI/IgA_registry/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels        WSI/IgA_registry/labels \\
        --val_csv followup_data/survival_validation_files.csv \\
        --output_dir results/survival_eval
"""

import argparse
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

# ---------------------------------------------------------------------------
# Resolve local torchmil package
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_torchmil_root = os.path.join(_project_root, "torchmil")
if (
    os.path.isdir(os.path.join(_torchmil_root, "torchmil"))
    and _torchmil_root not in sys.path
):
    sys.path.insert(0, _torchmil_root)

from torchmil.datasets import ProcessedMILDataset
from torchmil.data import collate_fn
from torchmil.models import abmil as abmil_module
from torchmil.models import deepgraphsurv as dgs_module
from torchmil.models import patch_gcn as patch_gcn_module

# ---------------------------------------------------------------------------
# Dataset loading (val-only)
# ---------------------------------------------------------------------------


def discover_bags(base_dir, extensions=(".npy", ".h5")):
    """Return relative bag paths (no extension) for flat or biopsy-nested layouts.

    Flat   : base_dir/slide.h5          → 'slide'
    Nested : base_dir/biopsy_nr/slide.h5 → 'biopsy_nr/slide'
    """
    bags = []
    for entry in os.scandir(base_dir):
        if entry.is_file():
            stem, ext = os.path.splitext(entry.name)
            if ext in extensions:
                bags.append(stem)
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file():
                    stem, ext = os.path.splitext(sub.name)
                    if ext in extensions:
                        bags.append(f"{entry.name}/{stem}")
    return sorted(bags)


def load_val_names(val_csv):
    if val_csv is None:
        return None
    raw = pd.read_csv(val_csv, header=None, dtype=str)
    col = raw.iloc[:, 0].str.strip()
    if col.iloc[0].lower() == "file_name":
        col = col.iloc[1:]
    return set(col)


def build_val_dataset(
    features_paths, labels_paths, coords_paths, bag_keys, dist_thr, val_csv
):
    val_names = load_val_names(val_csv)
    if val_names is None:
        raise ValueError("--val_csv is required for evaluation.")

    datasets = []
    for fp, lp, cp in zip(features_paths, labels_paths, coords_paths):
        labelled = set(discover_bags(lp, extensions=(".npy",)))
        available = discover_bags(fp)
        names_here = [n for n in available if n in val_names and n in labelled]
        if names_here:
            datasets.append(
                ProcessedMILDataset(
                    features_path=fp,
                    labels_path=lp,
                    coords_path=cp,
                    bag_keys=bag_keys,
                    dist_thr=dist_thr,
                    bag_names=names_here,
                )
            )
    if not datasets:
        raise RuntimeError(
            "No validation bags found — check --val_csv and directory paths."
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------


def _forward(model, batch, model_type):
    if model_type == "abmil":
        return model(batch["X"], batch["mask"])
    adj = batch["adj"]
    if adj.is_sparse:
        adj = adj.to_dense()
    return model(batch["X"], adj.float(), batch["mask"])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def get_risk_scores(model, loader, device, model_type):
    """Return (risk_scores, times, events) as float arrays, one entry per slide."""
    model.eval()
    risks, times, events = [], [], []
    # inference_mode disables autograd bookkeeping entirely — lower memory than no_grad
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            risk = _forward(model, batch, model_type).cpu().numpy()
            t = batch["Y"][:, 0].cpu().numpy()
            e = batch["Y"][:, 1].cpu().numpy()
            risks.extend(risk.tolist())
            times.extend(t.tolist())
            events.extend(e.tolist())
            del batch  # release GPU tensors immediately
            if device.type == "cuda" and i % 20 == 0:
                torch.cuda.empty_cache()  # defragment periodically
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(risks), np.array(times), np.array(events)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def concordance_index(risks, times, events, block=2000):
    """Harrell's C-index, block-vectorised with numpy.

    Processes `block` rows at a time so peak RAM stays at O(block × n) rather
    than O(n²).  For n=500 (typical val set) this uses ~2 MB; for n=5000 it
    uses ~200 MB.  Falls back gracefully to the full upper-triangle when n ≤ block.
    """
    n = len(risks)
    concordant = discordant = tied = 0

    for start in range(0, n, block):
        end = min(start + block, n)
        # (end-start, 1) vs (1, n) broadcasting
        ri = risks[start:end, None]
        rj = risks[None, :]
        ti = times[start:end, None]
        tj = times[None, :]
        ei = events[start:end, None]
        ej = events[None, :]

        # Only count each pair once (upper triangle: global col > global row)
        row_idx = np.arange(start, end)[:, None]
        col_idx = np.arange(n)[None, :]
        upper = col_idx > row_idx

        # Case A: row has shorter observed time and event
        mask_A = upper & (ti < tj) & (ei == 1)
        concordant += int(np.sum(mask_A & (ri > rj)))
        discordant += int(np.sum(mask_A & (ri < rj)))
        tied += int(np.sum(mask_A & (ri == rj)))

        # Case B: col has shorter observed time and event
        mask_B = upper & (tj < ti) & (ej == 1)
        concordant += int(np.sum(mask_B & (rj > ri)))
        discordant += int(np.sum(mask_B & (rj < ri)))
        tied += int(np.sum(mask_B & (rj == ri)))

    total = concordant + discordant + tied
    return concordant / total if total > 0 else 0.5


def kaplan_meier(times, events):
    """Return (t, S, ci_lower, ci_upper) with 95% confidence bands (Greenwood's formula).

    The KM estimator is a step function: S(t) = product over all event times
    <= t of (1 - d_i / n_i).  Greenwood's formula gives the pointwise variance,
    and the 95% CI is S ± 1.96 * S * sqrt(sum(d / n(n-d))).
    """
    if len(times) == 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
    order = np.argsort(times)
    t_sorted, e_sorted = times[order], events[order]
    t_plot, s_plot, ci_lo, ci_hi = [0.0], [1.0], [1.0], [1.0]
    S = 1.0
    greenwood_sum = 0.0
    n = len(t_sorted)
    i = 0
    while i < n:
        t_cur = t_sorted[i]
        j = i
        while j < n and t_sorted[j] == t_cur:
            j += 1
        at_risk = n - i
        deaths = int(e_sorted[i:j].sum())
        if deaths > 0:
            S *= 1.0 - deaths / at_risk
            denom = at_risk * (at_risk - deaths)
            if denom > 0:
                greenwood_sum += deaths / denom
            se = S * np.sqrt(greenwood_sum)
            t_plot.append(t_cur)
            s_plot.append(S)
            ci_lo.append(max(0.0, S - 1.96 * se))
            ci_hi.append(min(1.0, S + 1.96 * se))
        i = j
    return (np.array(t_plot), np.array(s_plot), np.array(ci_lo), np.array(ci_hi))


def log_rank_pvalue(times_a, events_a, times_b, events_b):
    """Two-sample log-rank test; returns p-value (chi-squared with 1 dof)."""
    from scipy.stats import chi2

    event_times = np.unique(
        np.concatenate([times_a[events_a == 1], times_b[events_b == 1]])
    )
    O1 = E1 = O2 = E2 = 0.0
    for t in event_times:
        n1 = np.sum(times_a >= t)
        n2 = np.sum(times_b >= t)
        d1 = int(np.sum((times_a == t) & (events_a == 1)))
        d2 = int(np.sum((times_b == t) & (events_b == 1)))
        n, d = n1 + n2, d1 + d2
        if n == 0:
            continue
        E1 += n1 * d / n
        E2 += n2 * d / n
        O1 += d1
        O2 += d2
    stat = 0.0
    if E1 > 0:
        stat += (O1 - E1) ** 2 / E1
    if E2 > 0:
        stat += (O2 - E2) ** 2 / E2
    return float(chi2.sf(stat, df=1))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

GROUP_COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]
GROUP_LABELS = ["Low risk", "Intermediate-low", "Intermediate-high", "High risk"]


def _assign_risk_groups(risks, n_groups):
    """Assign each slide to one of n_groups risk strata using rank-based quantiles.

    Using ranks instead of raw values avoids the np.digitize boundary problem
    where samples exactly equal to a quantile edge are assigned to the wrong bin,
    causing one group to be empty.
    """
    n = len(risks)
    sorted_idx = np.argsort(risks, kind="stable")
    group = np.empty(n, dtype=int)
    for g in range(n_groups):
        start = g * n // n_groups
        end = (g + 1) * n // n_groups if g < n_groups - 1 else n
        group[sorted_idx[start:end]] = g
    return group


def _survival_at_time(t_km, s_km, t_query):
    """Return S(t_query) by looking up the KM step function."""
    idx = np.searchsorted(t_km, t_query, side="right") - 1
    return s_km[max(0, idx)]


def plot_km_curves(
    risks, times, events, n_groups, output_dir, time_unit="days", at_risk_table=True
):
    """KM curves for n_groups equal-size risk strata.

    How to read the plot
    --------------------
    Each curve shows the probability of remaining event-free over time for one
    risk group.  A curve that drops steeply means frequent events; a flat curve
    means few events.

    - Solid line  : Kaplan-Meier estimate (the observed survival — this IS reality
                    for that group of slides).
    - Shaded band : 95% pointwise confidence interval (Greenwood's formula).
                    Wide bands mean few patients remain at risk; trust those
                    estimates less.
    - Tick marks  : each '|' on a curve marks a patient whose follow-up ended
                    without an event (censored).  Without them the curve looks
                    smoother than the data justify.
    - Log-rank p  : tests whether low-risk and high-risk curves differ beyond
                    chance.  Small p → the model separates the groups.
    - At-risk table (below x-axis): counts how many patients are still being
                    followed at each displayed time point.

    The model does not change any of these curves — it only decides which group
    each patient belongs to.  If the model is informative, the groups should have
    clearly separated curves.
    """
    group = _assign_risk_groups(risks, n_groups)

    # Layout: main plot + at-risk table rows
    n_table_rows = n_groups if at_risk_table else 0
    fig_h = 5 + 0.35 * n_table_rows
    fig, axes = (
        plt.subplots(
            2,
            1,
            figsize=(9, fig_h),
            gridspec_kw={"height_ratios": [5, 0.35 * n_table_rows + 0.01]},
        )
        if at_risk_table
        else (plt.subplots(1, 1, figsize=(9, 5)), [None])
    )
    ax = axes[0] if at_risk_table else axes

    # Choose representative time points for the at-risk table
    t_max = times.max()
    table_times = np.linspace(0, t_max, 6)

    for g in range(n_groups):
        mask = group == g
        t_g, e_g = times[mask], events[mask]
        t_km, s_km, ci_lo, ci_hi = kaplan_meier(t_g, e_g)

        label = GROUP_LABELS[g] if g < len(GROUP_LABELS) else f"Group {g}"
        label += f"  (n={mask.sum()}, events={int(e_g.sum())})"
        color = GROUP_COLORS[g % len(GROUP_COLORS)]

        # KM step curve
        ax.step(t_km, s_km, where="post", color=color, linewidth=2, label=label)
        # 95% CI shaded band — extend last value to right edge for aesthetics
        t_fill = np.append(t_km, t_max)
        lo_fill = np.append(ci_lo, ci_lo[-1])
        hi_fill = np.append(ci_hi, ci_hi[-1])
        ax.fill_between(t_fill, lo_fill, hi_fill, step="post", color=color, alpha=0.12)

        # Censoring tick marks
        censor_times = t_g[e_g == 0]
        censor_s = [_survival_at_time(t_km, s_km, tc) for tc in censor_times]
        ax.scatter(
            censor_times,
            censor_s,
            marker="|",
            color=color,
            s=40,
            linewidths=1.2,
            zorder=3,
        )

    # Log-rank p-value (lowest vs highest group)
    mask_lo = group == 0
    mask_hi = group == (n_groups - 1)
    p = log_rank_pvalue(
        times[mask_lo], events[mask_lo], times[mask_hi], events[mask_hi]
    )
    p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(
        0.97,
        0.97,
        f"Log-rank  low vs high\n{p_str}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )

    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    if not at_risk_table:
        ax.set_xlabel(f"Time ({time_unit})")

    # At-risk table
    if at_risk_table:
        ax_t = axes[1]
        ax_t.set_xlim(ax.get_xlim())
        ax_t.set_ylim(-0.5, n_groups - 0.5)
        ax_t.axis("off")
        ax_t.set_xlabel(f"Time ({time_unit})", labelpad=4)

        for g in range(n_groups - 1, -1, -1):  # top row = group 0
            mask = group == g
            t_g = times[mask]
            row_y = n_groups - 1 - g
            color = GROUP_COLORS[g % len(GROUP_COLORS)]
            label_short = GROUP_LABELS[g] if g < len(GROUP_LABELS) else f"Group {g}"
            ax_t.text(
                -0.01,
                row_y,
                label_short,
                transform=ax_t.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7,
                color=color,
            )
            for tt in table_times:
                n_at_risk = int(np.sum(t_g >= tt))
                ax_t.text(
                    tt,
                    row_y,
                    str(n_at_risk),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

        # Column headers
        for tt in table_times:
            ax_t.text(
                tt,
                n_groups - 0.1,
                f"{tt:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        fig.text(0.01, 0.02, "Number at risk", fontsize=7, color="gray", va="bottom")

    fig.tight_layout()
    path = os.path.join(output_dir, "km_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KM curves saved to {path}")


def plot_risk_distribution(risks, events, output_dir):
    """Histogram of risk scores, coloured by event status."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(risks.min(), risks.max(), 30)
    ax.hist(risks[events == 0], bins=bins, alpha=0.6, label="Censored", color="#2196F3")
    ax.hist(risks[events == 1], bins=bins, alpha=0.6, label="Event", color="#F44336")
    ax.set_xlabel("Predicted risk score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    path = os.path.join(output_dir, "risk_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Risk distribution saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MIL survival model: KM curves, C-index."
    )
    parser.add_argument(
        "--model_type", default="abmil", choices=["abmil", "deepgraphsurv", "patchgcn"]
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to saved model weights (.pth)."
    )
    parser.add_argument("--features_paths", nargs="+", required=True)
    parser.add_argument("--labels_paths", nargs="+", required=True)
    parser.add_argument(
        "--coords_paths",
        nargs="+",
        default=None,
        help="Required for deepgraphsurv / patchgcn.",
    )
    parser.add_argument(
        "--val_csv", required=True, help="CSV listing validation slide basenames."
    )
    parser.add_argument("--output_dir", default="results/survival_eval")
    parser.add_argument(
        "--n_groups",
        type=int,
        default=3,
        help="Number of risk strata for KM plot (default: 3).",
    )
    parser.add_argument(
        "--time_unit", default="days", help="Label for the time axis (default: days)."
    )
    parser.add_argument(
        "--no_at_risk_table",
        action="store_true",
        help="Omit the number-at-risk table below the KM plot.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size. Default 1 minimises peak GPU memory.",
    )
    parser.add_argument("--dist_thr", type=float, default=1.5)
    parser.add_argument(
        "--max_patches",
        type=int,
        default=40000,
        help="Patch subsample limit for graph models.",
    )
    parser.add_argument(
        "--cpu_inference",
        action="store_true",
        help="Run inference on CPU even when a GPU is present. "
        "Useful when GPU memory is insufficient for large slides.",
    )
    # Model architecture args (must match training)
    parser.add_argument("--att_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers_rep", type=int, default=1)
    parser.add_argument("--n_layers_att", type=int, default=1)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n_gcn_layers", type=int, default=4)
    parser.add_argument("--mlp_depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    is_graph = args.model_type in ("deepgraphsurv", "patchgcn")
    bag_keys = ["X", "Y", "adj", "coords"] if is_graph else ["X", "Y"]

    n = len(args.features_paths)
    coords_paths = args.coords_paths if args.coords_paths is not None else [None] * n
    if is_graph and args.coords_paths is None:
        parser.error(f"--coords_paths is required for {args.model_type}.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cpu_inference:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset -------------------------------------------------------------
    val_dataset = build_val_dataset(
        args.features_paths,
        args.labels_paths,
        coords_paths,
        bag_keys,
        args.dist_thr,
        args.val_csv,
    )
    print(f"Val bags: {len(val_dataset)}")

    # Self-contained patch subsampling (mirrors training scripts, no cross-import)
    base_collate = partial(collate_fn, sparse=not is_graph)

    def _make_collate(max_patches):
        if max_patches is None:
            return base_collate

        def _sub(bags):
            out = []
            for bag in bags:
                n_p = bag["X"].shape[0]
                if n_p > max_patches:
                    idx = torch.randperm(n_p)[:max_patches]
                    new_bag = {}
                    for k, v in bag.items():
                        if k in ("X", "coords") and isinstance(v, torch.Tensor):
                            new_bag[k] = v[idx]
                        elif k == "adj" and isinstance(v, torch.Tensor):
                            if v.is_sparse:
                                v = v.coalesce()
                                row, col, vals = (
                                    v.indices()[0],
                                    v.indices()[1],
                                    v.values(),
                                )
                                m = torch.full((v.size(0),), -1, dtype=torch.long)
                                m[idx] = torch.arange(len(idx))
                                keep = (m[row] >= 0) & (m[col] >= 0)
                                new_bag[k] = torch.sparse_coo_tensor(
                                    torch.stack([m[row[keep]], m[col[keep]]]),
                                    vals[keep],
                                    (len(idx), len(idx)),
                                )
                            else:
                                new_bag[k] = v[idx][:, idx]
                        else:
                            new_bag[k] = v
                    bag = new_bag
                out.append(bag)
            return base_collate(out)

        return _sub

    _collate = _make_collate(args.max_patches)

    # num_workers=0: avoids spawning RAM-heavy worker processes for a one-shot eval
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    # --- Model ---------------------------------------------------------------
    feat_dim = int(val_dataset[0]["X"].shape[-1])
    if args.model_type == "abmil":
        model = abmil_module.ABMIL(att_dim=args.att_dim)
    elif args.model_type == "deepgraphsurv":
        model = dgs_module.DeepGraphSurv(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            hidden_dim=args.hidden_dim,
            n_layers_rep=args.n_layers_rep,
            n_layers_att=args.n_layers_att,
            K=args.K,
            dropout=args.dropout,
            compute_lambda_max=False,
        )
    else:
        model = patch_gcn_module.PatchGCN(
            in_shape=(feat_dim,),
            n_gcn_layers=args.n_gcn_layers,
            mlp_depth=args.mlp_depth,
            hidden_dim=args.hidden_dim,
            att_dim=args.att_dim,
            dropout=args.dropout,
        )

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Inference -----------------------------------------------------------
    risks, times, events = get_risk_scores(model, loader, device, args.model_type)
    # Release the model from GPU as soon as inference is done
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Slides evaluated: {len(risks)}  |  events: {int(events.sum())}")

    c_index = concordance_index(risks, times, events)
    print(f"C-index: {c_index:.4f}")

    # --- Save risk scores CSV ------------------------------------------------
    csv_path = os.path.join(args.output_dir, "risk_scores.csv")
    pd.DataFrame({"risk": risks, "time": times, "event": events}).to_csv(
        csv_path, index=False
    )
    print(f"Risk scores saved to {csv_path}")

    # --- Plots ---------------------------------------------------------------
    plot_km_curves(
        risks,
        times,
        events,
        n_groups=args.n_groups,
        output_dir=args.output_dir,
        time_unit=args.time_unit,
        at_risk_table=not args.no_at_risk_table,
    )
    plot_risk_distribution(risks, events, args.output_dir)


if __name__ == "__main__":
    main()
