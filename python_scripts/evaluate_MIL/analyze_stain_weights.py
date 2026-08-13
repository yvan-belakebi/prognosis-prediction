"""
analyze_stain_weights.py — What did the multi-stain model actually use each stain for?

Consumes a checkpoint from multistain_MIL.py and answers three questions:

1. **How much did each stain move each prediction?**  The head is
   ``LayerNorm -> Dropout -> Linear`` on top of the attention-pooled bag vector
   ``z = sum_s a_s t_s``, so the output decomposes *exactly* into per-stain terms::

       contribution_s = (W . gamma / sigma) . (a_s * t_s)
       prediction     = sum_s contribution_s + const

   This is signed — it separates "the model looked at Congo" from "Congo pushed
   this patient's risk up" — which the raw attention weight ``a_s`` cannot do.
   The script asserts the identity holds on every batch, so a future change to the
   head shape fails loudly instead of silently producing wrong attributions.

   Caveat: LayerNorm's mu and sigma depend on all stains jointly, so this
   apportions the *realised* score. It is not a counterfactual — that is (2).

2. **What happens if a stain is taken away?**  Leave-one-stain-out: re-run with the
   stain masked and report the change in C-index (survival) or MAE (regression).
   Biopsies that would be left with no stain at all are held out of that stain's
   ablation. If the model was trained with ``--stain_dropout > 0``, masking at
   inference is in-distribution and this measures reliance rather than shift.

3. **Is the stain panel itself prognostic?**  Which stains a biopsy has is not
   random — Congo gets ordered when amyloid is suspected. The script scores the
   availability pattern *alone* (no image data) so you can see how much apparent
   stain signal is really panel-composition signal.

Outputs (in --output_dir):

    stain_contributions.csv          one row per biopsy: prediction, label,
                                     per-stain contribution / attention / lift
    stain_contribution_heatmap.png   biopsies (sorted by prediction) x stains
    stain_contribution_by_stain.png  cohort distribution per stain
    stain_waterfall.png              three example biopsies, low / median / high
    stain_ablation.csv/.png          leave-one-stain-out metric change
    panel_baseline.csv               availability-only prognostic baseline

Usage:
    python python_scripts/evaluate_MIL/analyze_stain_weights.py \\
        --checkpoint     checkpoints_multistain/multistain_survival_best.pth \\
        --features_paths WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested \\
        --labels_paths   WSI/IgA/trident/labels \\
        --stain_csvs     label_csvs/labels_unfiltered.csv \\
        --val_csv        validation_files_csvs/survival_validation_files.csv \\
        --split val --output_dir results/stain_analysis

The stain vocabulary and architecture come from multistain_config.json next to the
checkpoint (override with --config), so the flags above only describe the data.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve the local torchmil package and the MIL scripts
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_torchmil_root = os.path.join(_project_root, "torchmil")
if (
    os.path.isdir(os.path.join(_torchmil_root, "torchmil"))
    and _torchmil_root not in sys.path
):
    sys.path.insert(0, _torchmil_root)

_mil_dir = os.path.abspath(os.path.join(_script_dir, "..", "MIL"))
if _mil_dir not in sys.path:
    sys.path.insert(0, _mil_dir)

from mil_utils import load_val_names, load_authorized_slides  # noqa: E402
from multistain_data import (  # noqa: E402
    MultiStainBiopsyDataset,
    StainVocabulary,
    index_biopsies,
    split_records,
)
from multistain_model import MultiStainFusion  # noqa: E402
from evaluate_survival import concordance_index  # noqa: E402


# ---------------------------------------------------------------------------
# Palette
#
# Values are taken verbatim from the validated reference palette rather than
# re-stepped here: diverging = blue <-> red (warm/cool poles that read as
# opposite) over a neutral gray midpoint, so the midpoint reads as "no effect".
# Magnitude-only panels use the single sequential hue (blue). Text stays in ink
# tokens; a mark beside it carries the identity.
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "neg": "#2a78d6",  # cool pole — pushes the prediction down
        "pos": "#e34948",  # warm pole — pushes the prediction up
        "mid": "#f0efec",  # neutral midpoint
        "absent": "#e1e0d9",  # stain not available (hatched, never a ramp value)
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "neg": "#3987e5",
        "pos": "#e66767",
        "mid": "#383835",
        "absent": "#2c2c2a",
        "series": "#3987e5",
    },
}


def diverging_cmap(theme: dict) -> LinearSegmentedColormap:
    """Blue -> neutral gray -> red, equal steps per arm, gray at the midpoint."""
    cmap = LinearSegmentedColormap.from_list(
        "stain_diverging", [theme["neg"], theme["mid"], theme["pos"]], N=256
    )
    cmap.set_bad(theme["absent"])
    return cmap


def style_axes(ax, theme: dict, grid_axis="both"):
    """Recessive chrome: hairline grid behind the marks, no top/right spines."""
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=theme["grid"], linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)


def new_figure(theme: dict, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    fig.patch.set_facecolor(theme["surface"])
    return fig, ax


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------
def build_model_from_config(cfg: dict) -> MultiStainFusion:
    return MultiStainFusion(
        in_dim=cfg["feat_dim"],
        n_stains=len(cfg["stain_names"]),
        d_model=cfg["d_model"],
        stain_layers=cfg["stain_layers"],
        agg_layers=cfg["agg_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        pool_att_dim=cfg["pool_att_dim"],
        gated=cfg["gated"],
        out_dim=1,
        share_stain_encoder=cfg["share_stain_encoder"],
    )


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------
def _head_parts(head: nn.Module):
    """Return the (LayerNorm, Linear) pair of the prediction head."""
    ln = next((m for m in head.modules() if isinstance(m, nn.LayerNorm)), None)
    lin = next((m for m in head.modules() if isinstance(m, nn.Linear)), None)
    if ln is None or lin is None:
        raise RuntimeError(
            "Expected the head to contain a LayerNorm and a Linear layer; "
            "the exact per-stain decomposition does not apply to this head."
        )
    return ln, lin


def decompose(head: nn.Module, a: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
    """Split the prediction into per-stain contributions.

    Arguments:
        head: The model's ``LayerNorm -> Dropout -> Linear`` head.
        a: Stain attention weights (already softmaxed over present stains), `(B, S)`.
        t: Stain tokens entering the pooling, `(B, S, d)`.
        mask: Stain availability, `(B, S)`.

    Returns:
        contrib: `(B, S)` signed contributions, zero where the stain is absent.
        const: `(B,)` offset such that ``contrib.sum(1) + const`` is the prediction.
    """
    ln, lin = _head_parts(head)
    z = (a.unsqueeze(-1) * t).sum(1)  # (B, d) — the pooled bag vector
    mu = z.mean(-1, keepdim=True)
    var = z.var(-1, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + ln.eps)

    w = lin.weight[0]  # (d,) — out_dim is 1
    gamma = ln.weight if ln.weight is not None else torch.ones_like(w)
    beta = ln.bias if ln.bias is not None else torch.zeros_like(w)

    scale = w * gamma * inv  # (B, d)
    contrib = torch.einsum("bd,bsd->bs", scale, a.unsqueeze(-1) * t) * mask
    const = (w * beta).sum() - scale.sum(-1) * mu.squeeze(-1)
    if lin.bias is not None:
        const = const + lin.bias[0]
    return contrib, const


@torch.no_grad()
def collect(model, loader, device, check_tol=1e-4):
    """Run the model and return predictions, contributions, attention and masks."""
    captured = {}
    handle = model.stain_pool.register_forward_pre_hook(
        lambda _mod, args: captured.__setitem__("t", args[0])
    )
    preds, contribs, atts, masks = [], [], [], []
    worst = 0.0
    try:
        for batch in loader:
            X = batch["X"].to(device)
            mask = batch["stain_mask"].to(device)
            pred, stain_att, _ = model(X, mask, return_att=True)

            a = torch.softmax(stain_att.masked_fill(~mask, -torch.inf), dim=1)
            contrib, const = decompose(model.head, a, captured["t"], mask)

            # The decomposition is exact by construction; verify it so a future
            # change to the head fails here instead of silently mis-attributing.
            worst = max(worst, float((pred - contrib.sum(1) - const).abs().max()))

            preds.append(pred.cpu())
            contribs.append(contrib.cpu())
            atts.append(a.cpu())
            masks.append(mask.cpu())
    finally:
        handle.remove()

    if worst > check_tol:
        raise RuntimeError(
            f"Per-stain contributions do not reconstruct the prediction "
            f"(max error {worst:.3g} > {check_tol:g}). The head is no longer "
            f"LayerNorm -> Linear over the pooled vector, so the attribution "
            f"formula in decompose() needs updating."
        )
    print(f"  decomposition self-check: max |prediction - sum(contrib)| = {worst:.2e}")
    return (
        torch.cat(preds).numpy(),
        torch.cat(contribs).numpy(),
        torch.cat(atts).numpy(),
        torch.cat(masks).numpy(),
    )


@torch.no_grad()
def predict_with_mask(model, loader, device, drop_stain=None):
    """Predictions, optionally with one stain masked out.

    Biopsies whose only stain is ``drop_stain`` keep it — an empty stain set has no
    defined prediction — and are flagged so the caller can exclude them.
    """
    preds, usable = [], []
    for batch in loader:
        X = batch["X"].to(device)
        mask = batch["stain_mask"].to(device)
        keep = torch.ones(mask.shape[0], dtype=torch.bool, device=device)
        if drop_stain is not None:
            can_drop = mask[:, drop_stain] & (mask.sum(1) > 1)
            mask = mask.clone()
            mask[can_drop, drop_stain] = False
            keep = can_drop
        preds.append(model(X, mask).cpu())
        usable.append(keep.cpu())
    return torch.cat(preds).numpy(), torch.cat(usable).numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(task, preds, labels, subset=None):
    """C-index (survival, higher better) or MAE (regression, lower better)."""
    if subset is not None:
        preds, labels = preds[subset], labels[subset]
    if len(preds) < 2:
        return float("nan")
    if task == "survival":
        return concordance_index(preds, labels[:, 0], labels[:, 1])
    return float(np.abs(preds - labels[:, 0]).mean())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_heatmap(df, stain_names, theme, path, task, label_name):
    """Biopsies (rows, sorted by prediction) x stains (columns), signed contribution.

    The one figure that shows whether high-prediction biopsies are driven by a
    particular stain, with the missingness pattern visible at the same time.
    """
    order = np.argsort(df["prediction"].to_numpy())
    M = df[[f"contrib_{s}" for s in stain_names]].to_numpy()[order]
    present = df[[f"present_{s}" for s in stain_names]].to_numpy().astype(bool)[order]
    M = np.ma.masked_array(M, mask=~present)

    # Symmetric limits at a robust quantile so one outlier does not flatten the rest.
    lim = float(np.nanpercentile(np.abs(M.compressed()), 98)) if M.count() else 1.0
    lim = max(lim, 1e-6)

    n_rows = len(df)
    fig, ax = new_figure(theme, figsize=(1.15 * len(stain_names) + 4, min(0.16 * n_rows + 2.2, 14)))
    mesh = ax.pcolormesh(
        M,
        cmap=diverging_cmap(theme),
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        edgecolors=theme["surface"],
        linewidth=1.2,
    )
    # Absent stains get a texture, never a ramp value — "no data" must not read
    # as "no effect" (which is the neutral midpoint of the same gray family).
    for r, c in zip(*np.where(~present)):
        ax.add_patch(
            plt.Rectangle(
                (c, r), 1, 1, facecolor=theme["absent"], edgecolor=theme["surface"],
                linewidth=1.2, hatch="///", alpha=0.65,
            )
        )

    ax.set_xticks(np.arange(len(stain_names)) + 0.5)
    ax.set_xticklabels(stain_names, rotation=30, ha="right", color=theme["ink_secondary"])
    ax.set_ylabel(
        f"biopsies, sorted by predicted {'risk' if task == 'survival' else label_name}",
        color=theme["ink_secondary"], fontsize=10,
    )
    if n_rows <= 45:
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(df["biopsy"].to_numpy()[order], fontsize=7, color=theme["muted"])
    else:
        ax.set_yticks([])
    style_axes(ax, theme, grid_axis="none")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label(
        "contribution to the prediction  (← lowers   raises →)",
        color=theme["ink_secondary"], fontsize=9,
    )
    cbar.ax.tick_params(colors=theme["muted"], labelsize=8)
    cbar.outline.set_visible(False)

    ax.legend(
        handles=[Patch(facecolor=theme["absent"], hatch="///", label="stain not available")],
        loc="upper left", bbox_to_anchor=(0, -0.06), frameon=False,
        fontsize=9, labelcolor=theme["ink_secondary"],
    )
    ax.set_title(
        "Per-stain contribution to each biopsy's prediction",
        color=theme["ink"], fontsize=12, pad=12, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    print(f"  wrote {path}")


def plot_by_stain(df, stain_names, theme, path):
    """Cohort distribution of contribution per stain, ordered by mean magnitude.

    One hue for every box: the stains are nominal, and the axis labels already
    carry their identity, so colour would only re-encode the ordering.
    """
    data, labels = [], []
    for s in stain_names:
        v = df.loc[df[f"present_{s}"] == 1, f"contrib_{s}"].to_numpy()
        if len(v):
            data.append(v)
            labels.append(f"{s}  (n={len(v)})")
    order = np.argsort([-np.abs(d).mean() for d in data])
    data = [data[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = new_figure(theme, figsize=(8, 0.55 * len(data) + 2.4))
    bp = ax.boxplot(
        data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
        medianprops=dict(color=theme["ink"], linewidth=2),
        whiskerprops=dict(color=theme["axis"], linewidth=1),
        capprops=dict(color=theme["axis"], linewidth=1),
        boxprops=dict(facecolor=theme["series"], edgecolor=theme["series"], alpha=0.28, linewidth=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.scatter(
            v, i + rng.uniform(-0.13, 0.13, size=len(v)),
            s=9, color=theme["series"], alpha=0.45, linewidths=0, zorder=3,
        )
    ax.axvline(0, color=theme["axis"], linewidth=1.5, zorder=1)
    ax.set_yticklabels(labels, color=theme["ink_secondary"])
    ax.set_xlabel("contribution to the prediction", color=theme["ink_secondary"], fontsize=10)
    ax.set_title(
        "Which stains move the prediction, and in which direction",
        color=theme["ink"], fontsize=12, pad=12, loc="left",
    )
    style_axes(ax, theme, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    print(f"  wrote {path}")


def plot_waterfall(df, stain_names, theme, path, task, label_name):
    """Three example biopsies — lowest, median and highest prediction.

    The three panels share one x-scale. Letting each auto-scale would size a 0.04
    contribution like a 0.6 one and make the panels look comparable when they are
    not — the small-multiples version of the two-y-axis mistake.
    """
    order = np.argsort(df["prediction"].to_numpy())
    picks = [
        ("lowest", order[0]),
        ("median", order[len(order) // 2]),
        ("highest", order[-1]),
    ]
    cols = [f"contrib_{s}" for s in stain_names]
    shared = max(float(np.abs(df.iloc[[i for _, i in picks]][cols].to_numpy()).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(stain_names) + 3.4))
    fig.patch.set_facecolor(theme["surface"])

    for ax, (tag, i) in zip(np.atleast_1d(axes), picks):
        row = df.iloc[i]
        vals, names = [], []
        for s in stain_names:
            if row[f"present_{s}"] == 1:
                vals.append(row[f"contrib_{s}"])
                names.append(s)
        idx = np.argsort(vals)
        vals = np.array(vals)[idx]
        names = [names[k] for k in idx]

        colors = [theme["pos"] if v >= 0 else theme["neg"] for v in vals]
        ax.barh(np.arange(len(vals)), vals, height=0.62, color=colors, linewidth=0)
        ax.axvline(0, color=theme["axis"], linewidth=1.5)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(names, color=theme["ink_secondary"], fontsize=9)

        for k, v in enumerate(vals):
            ax.text(
                v + np.sign(v) * shared * 0.04, k, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=theme["ink_secondary"],
            )
        ax.set_xlim(-shared * 1.35, shared * 1.35)

        if task == "survival":
            outcome = "event" if row["event"] == 1 else "censored"
            sub = f"risk {row['prediction']:+.2f}  ·  {outcome} at {row['time']:.0f} d"
        else:
            sub = f"predicted {row['prediction']:.1f}  ·  observed {row['target']:.1f} {label_name}"
        ax.set_title(
            f"{tag}: {row['biopsy']}\n{sub}",
            color=theme["ink"], fontsize=10, pad=10, loc="left",
        )
        style_axes(ax, theme, grid_axis="x")

    handles = [
        Patch(facecolor=theme["pos"], label="raises the prediction"),
        Patch(facecolor=theme["neg"], label="lowers the prediction"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False,
        fontsize=9, labelcolor=theme["ink_secondary"], bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    print(f"  wrote {path}")


def plot_ablation(abl, theme, path, task):
    """Leave-one-stain-out change in the validation metric."""
    abl = abl.dropna(subset=["delta"]).sort_values("delta")
    if abl.empty:
        print("  ablation: nothing to plot")
        return
    metric = "C-index" if task == "survival" else "MAE"
    colors = [theme["pos"] if d >= 0 else theme["neg"] for d in abl["delta"]]

    fig, ax = new_figure(theme, figsize=(8, 0.5 * len(abl) + 2.6))
    ax.barh(np.arange(len(abl)), abl["delta"], height=0.6, color=colors, linewidth=0)
    ax.axvline(0, color=theme["axis"], linewidth=1.5)
    ax.set_yticks(np.arange(len(abl)))
    ax.set_yticklabels(
        [f"{r.stain}  (n={int(r.n)})" for r in abl.itertuples()],
        color=theme["ink_secondary"],
    )
    span = max(abl["delta"].abs().max(), 1e-6)
    for k, v in enumerate(abl["delta"]):
        ax.text(
            v + np.sign(v) * span * 0.04, k, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8, color=theme["ink_secondary"],
        )
    ax.set_xlim(-span * 1.4, span * 1.4)
    ax.set_xlabel(
        f"loss of {metric} when the stain is removed", color=theme["ink_secondary"], fontsize=10
    )
    ax.set_title(
        "Leave-one-stain-out: how much does the model rely on each stain?",
        color=theme["ink"], fontsize=12, pad=12, loc="left",
    )
    ax.legend(
        handles=[
            Patch(facecolor=theme["pos"], label="removing it hurts (model relies on it)"),
            Patch(facecolor=theme["neg"], label="removing it helps"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False,
        fontsize=9, labelcolor=theme["ink_secondary"],
    )
    style_axes(ax, theme, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Availability-only baseline
# ---------------------------------------------------------------------------
def _strength(task, v, labels):
    """How much does predictor ``v`` alone say about the outcome? Higher = more.

    Survival: a tie-aware C-index, folded to the informative direction.  Two
    corrections are needed before a 0/1 availability flag can be compared with the
    model's own C-index:

    * ``evaluate_survival.concordance_index`` puts tied-risk pairs in the
      denominator without crediting them.  That is immaterial for continuous risks
      but not for an indicator, where almost every pair ties — both directions then
      score far below 0.5 and nothing looks predictive.  Harrell's convention
      credits ties 0.5, which is recovered exactly from the two directional scores:
      ``(C + 0.5T)/total  ==  0.5 + (c_forward - c_backward) / 2``
      (using ``C + D + T == total``).  For a tie-free predictor ``c_backward`` is
      ``1 - c_forward`` and this collapses back to ``c_forward``.
    * An indicator is equally informative reversed — 'has Congo' predicting *low*
      risk is still panel signal — so the score is folded to its better direction
      and the sign reported separately.

    Regression: |Pearson r|, since a 0/1 indicator used directly as a predicted
    label value would have a meaningless MAE.
    """
    if task == "survival":
        forward = concordance_index(v, labels[:, 0], labels[:, 1])
        backward = concordance_index(-v, labels[:, 0], labels[:, 1])
        tie_aware = 0.5 + (forward - backward) / 2.0
        return (tie_aware, +1) if tie_aware >= 0.5 else (1.0 - tie_aware, -1)
    if np.std(v) == 0:
        return float("nan"), 0
    return abs(float(np.corrcoef(v, labels[:, 0])[0, 1])), 0


def panel_baseline(task, masks, labels, preds, stain_names):
    """Score the stain-availability pattern alone, with no image data.

    Which stains a biopsy has is not random — Congo gets ordered when amyloid is
    suspected — so panel composition can be prognostic by itself. If it is, apparent
    per-stain importance is partly panel-composition signal and has to be
    interpreted conditional on the panel. When scikit-survival is installed a
    multivariate Cox fit on the whole indicator matrix is added.
    """
    rows = []
    for i, name in enumerate(stain_names):
        v = masks[:, i].astype(float)
        if 0 < v.sum() < len(v):
            s, d = _strength(task, v, labels)
            rows.append({"predictor": f"has {name}", "strength": s, "direction": d})
    s, d = _strength(task, masks.sum(1).astype(float), labels)
    rows.append({"predictor": "number of stains", "strength": s, "direction": d})

    if task == "survival":
        try:
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            y = np.array(
                [(bool(e), float(t)) for t, e in labels[:, :2]],
                dtype=[("event", bool), ("time", float)],
            )
            X = masks.astype(float)
            fit = CoxPHSurvivalAnalysis(alpha=0.1).fit(X, y)
            s, d = _strength(task, fit.predict(X), labels)
            rows.append(
                {"predictor": "availability pattern (multivariate Cox)",
                 "strength": s, "direction": d}
            )
        except Exception as exc:  # sksurv missing, or a degenerate fit
            print(f"  panel baseline: skipping multivariate Cox ({type(exc).__name__}: {exc})")

    s, d = _strength(task, preds, labels)
    rows.append({"predictor": "the model (for reference)", "strength": s, "direction": d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyse per-stain contributions of a trained multi-stain model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Trained .pth from multistain_MIL.py.")
    parser.add_argument(
        "--config",
        default=None,
        help="multistain_config.json (default: next to the checkpoint).",
    )
    parser.add_argument("--features_paths", nargs="+", required=True)
    parser.add_argument("--labels_paths", nargs="+", required=True)
    parser.add_argument("--stain_csvs", nargs="+", required=True)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "all"],
        help="Which biopsies to analyse. 'val' needs --val_csv.",
    )
    parser.add_argument("--authorized_slides_csv", default=None)
    parser.add_argument("--file_ext", default=".h5", choices=[".h5", ".npy"])
    parser.add_argument("--min_stains", type=int, default=1)
    parser.add_argument(
        "--patches_per_stain",
        type=int,
        default=None,
        help="Override the training value. Raising it gives the encoders more "
             "context than they saw in training, which shifts the scores.",
    )
    parser.add_argument("--output_dir", default="results/stain_analysis")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--theme", default="light", choices=["light", "dark"])
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "multistain_config.json"
    )
    if not os.path.isfile(cfg_path):
        parser.error(f"No config at {cfg_path} — pass --config explicitly.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    n_cohorts = len(args.features_paths)
    if len(args.labels_paths) != n_cohorts:
        parser.error("--features_paths and --labels_paths must have the same length.")
    if len(args.stain_csvs) == 1:
        args.stain_csvs = args.stain_csvs * n_cohorts
    if args.split == "val" and args.val_csv is None:
        parser.error("--split val requires --val_csv.")

    task = cfg["task"]
    label_name = cfg.get("label_name", "label")
    label_mean = float(cfg.get("label_mean", 0.0))
    label_std = float(cfg.get("label_std", 1.0))
    stain_names = cfg["stain_names"]
    theme = THEMES[args.theme]
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  task: {task}  |  stains: {stain_names}")
    if cfg.get("stain_dropout", 0.0) == 0.0 and not args.skip_ablation:
        print(
            "  Note: this model was trained with --stain_dropout 0, so masking a stain "
            "at inference is out of distribution; read the ablation as an upper bound "
            "on reliance."
        )

    # --- Data ----------------------------------------------------------------
    vocab = StainVocabulary(cfg["stains"])
    records = []
    for fp, lp, sc in zip(args.features_paths, args.labels_paths, args.stain_csvs):
        records += index_biopsies(
            fp, lp, sc, vocab,
            file_ext=args.file_ext,
            authorized_slides=load_authorized_slides(args.authorized_slides_csv),
            min_stains=args.min_stains,
        )
    train_recs, val_recs = split_records(records, load_val_names(args.val_csv))
    records = {"val": val_recs, "train": train_recs, "all": records}[args.split]
    if not records:
        parser.error(f"No biopsies in the '{args.split}' split.")
    print(f"Analysing {len(records)} biopsies ({args.split} split)")

    ds = MultiStainBiopsyDataset(
        records,
        n_stains=len(stain_names),
        patches_per_stain=args.patches_per_stain or cfg["patches_per_stain"],
        random_subsample=False,
        feat_dim=cfg["feat_dim"],
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --- Contributions --------------------------------------------------------
    preds, contribs, atts, masks = collect(model, loader, device)
    # Model outputs are in normalised units for regression; put everything back
    # into raw label units so the numbers match the target the reader knows.
    preds_raw = preds * label_std + label_mean
    contribs = contribs * label_std

    labels = np.stack([np.asarray(r["label"], dtype=float)[:2] if task == "survival"
                       else np.asarray(r["label"], dtype=float)[:1] for r in records])
    if task == "regression":
        labels = labels.reshape(-1, 1)

    n_present = masks.sum(1, keepdims=True)
    df = pd.DataFrame({"biopsy": [r["biopsy"] for r in records], "prediction": preds_raw})
    if task == "survival":
        df["time"] = labels[:, 0]
        df["event"] = labels[:, 1]
    else:
        df["target"] = labels[:, 0]
    df["n_stains"] = n_present[:, 0]
    for i, s in enumerate(stain_names):
        df[f"present_{s}"] = masks[:, i].astype(int)
        df[f"contrib_{s}"] = np.where(masks[:, i], contribs[:, i], np.nan)
        df[f"att_{s}"] = np.where(masks[:, i], atts[:, i], np.nan)
        # Attention is normalised over the stains a biopsy happens to have, so the
        # raw weight is not comparable across different panel sizes; lift is.
        df[f"lift_{s}"] = df[f"att_{s}"] * n_present[:, 0]
    csv_path = os.path.join(args.output_dir, "stain_contributions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    plot_df = df.fillna({f"contrib_{s}": 0.0 for s in stain_names})
    plot_heatmap(plot_df, stain_names, theme,
                 os.path.join(args.output_dir, "stain_contribution_heatmap.png"), task, label_name)
    plot_by_stain(plot_df, stain_names, theme,
                  os.path.join(args.output_dir, "stain_contribution_by_stain.png"))
    plot_waterfall(plot_df, stain_names, theme,
                   os.path.join(args.output_dir, "stain_waterfall.png"), task, label_name)

    # --- Leave-one-stain-out --------------------------------------------------
    if not args.skip_ablation:
        print("Leave-one-stain-out ablation:")
        rows = []
        for i, name in enumerate(stain_names):
            abl_preds, usable = predict_with_mask(model, loader, device, drop_stain=i)
            if usable.sum() < 2:
                rows.append({"stain": name, "n": int(usable.sum()), "full": np.nan,
                             "ablated": np.nan, "delta": np.nan, "mean_abs_shift": np.nan})
                continue
            full = score(task, preds, labels, subset=usable)
            abl = score(task, abl_preds, labels, subset=usable)
            # Positive delta always means "removing this stain made the model worse".
            delta = (full - abl) if task == "survival" else (abl - full)
            shift = float(np.abs((abl_preds - preds)[usable]).mean() * label_std)
            rows.append({"stain": name, "n": int(usable.sum()), "full": full,
                         "ablated": abl, "delta": delta, "mean_abs_shift": shift})
            print(f"  {name:>18s}  n={usable.sum():4d}  full={full:.4f}  "
                  f"without={abl:.4f}  delta={delta:+.4f}  |shift|={shift:.3f}")
        abl_df = pd.DataFrame(rows)
        abl_path = os.path.join(args.output_dir, "stain_ablation.csv")
        abl_df.to_csv(abl_path, index=False)
        print(f"  wrote {abl_path}")
        plot_ablation(abl_df, theme, os.path.join(args.output_dir, "stain_ablation.png"), task)

    # --- Availability-only baseline ------------------------------------------
    unit = "best-direction C-index" if task == "survival" else "|Pearson r|"
    chance = 0.5 if task == "survival" else 0.0
    print(f"Availability-only baseline, no image data ({unit}, chance = {chance}):")
    base = panel_baseline(task, masks, labels, preds, stain_names)
    base_path = os.path.join(args.output_dir, "panel_baseline.csv")
    base.to_csv(base_path, index=False)
    print(base.to_string(index=False))
    print(f"  wrote {base_path}")

    panel_only = base[~base["predictor"].str.startswith("the model")]["strength"]
    best = float(panel_only.max()) if len(panel_only) else float("nan")
    if np.isfinite(best) and best > (0.60 if task == "survival" else 0.30):
        print(
            f"  WARNING: stain availability alone reaches {best:.3f} ({unit}) — which "
            f"stains a biopsy has is prognostic here on its own, so per-stain "
            f"importance has to be read conditional on the panel, not as evidence "
            f"about the tissue in that stain."
        )


if __name__ == "__main__":
    main()
