"""
multistain_MIL.py — Train the UNICORN-style multi-stain fusion model on biopsies.

One sample is a *biopsy*, not a slide: every slide of the biopsy is grouped by its
stain (from the define_labels.py CSV), each stain gets its own 2-layer transformer,
and a second 2-layer transformer aggregates the per-stain tokens into one biopsy
vector.  A single linear head then predicts either a Cox risk score (--task survival)
or a continuous target (--task regression).

    multistain_data.py   biopsy-level, multi-stain bags
    multistain_model.py  MultiStainFusion (per-stain encoders + aggregator + head)
    multistain_MIL.py    this training script

Usage (survival):
    python python_scripts/MIL/multistain_MIL.py --task survival \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels \\
        --labels_csv    label_csvs/labels.csv \\
        --stains PAS PASM "HES|HE" Congo AFOG \\
        --patches_per_stain 512 --batch_size 8 --epochs 50 \\
        --checkpoint_dir checkpoints_multistain --log_dir results/losses_multistain

Usage (regression, eGFR):
    python python_scripts/MIL/multistain_MIL.py --task regression \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels_regression \\
        --labels_csv    label_csvs/labels_regression.csv \\
        --stains PAS PASM "HES|HE" Congo AFOG

The train/val split is the labels CSV's split column.

A stain argument may list aliases separated by '|' ("HES|HE"); the first name is the
one reported.  Stains absent from --stains are ignored, and biopsies keep whichever
of the selected stains they happen to have — missing stains are masked out.

Regression labels are z-scored on the training biopsies before training; the reported
MAE and the residual plot (``residuals.png`` in the log dir) are converted back to raw
label units, and ``label_mean`` / ``label_std`` go into the config so inference can
invert the model output with ``raw = pred * label_std + label_mean``.

Alongside the checkpoints, ``multistain_config.json`` records the stain vocabulary and
the architecture, so evaluation and inference can rebuild the exact same model.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve local torchmil package (same convention as MIL.py / regression_MIL.py)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_torchmil_root = os.path.join(_project_root, "torchmil")
if (
    os.path.isdir(os.path.join(_torchmil_root, "torchmil"))
    and _torchmil_root not in sys.path
):
    sys.path.insert(0, _torchmil_root)

_evaluate_dir = os.path.abspath(os.path.join(_script_dir, "..", "evaluate_MIL"))
if _evaluate_dir not in sys.path:
    sys.path.insert(0, _evaluate_dir)

from mil_utils import (  # noqa: E402
    EventStratifiedBatchSampler,
    LossLogger,
    load_authorized_slides,
    load_labels_table,
    load_last_checkpoint,
    save_last_checkpoint,
)
from multistain_data import (  # noqa: E402
    MultiStainBiopsyDataset,
    StainVocabulary,
    index_biopsies,
    split_records,
)
from multistain_model import (  # noqa: E402
    EarlyFusionMIL,
    FlexMoEFusion,
    MultiStainFusion,
    build_combos,
)
from MIL import cox_ph_loss  # noqa: E402
from evaluate_survival import concordance_index  # noqa: E402


class MultiStainLogger(LossLogger):
    """Loss log with one extra metric column (C-index for survival, MAE for
    regression), plotted in a second panel next to the losses."""

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_metric"]

    def __init__(self, log_dir, metric_name, resume_epoch=None):
        super().__init__(log_dir, resume_epoch=resume_epoch)
        self.metric_name = metric_name

    def log(self, epoch, train_loss, val_loss=None, val_metric=None):
        self._append(
            [
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_metric is None else f"{val_metric:.4f}",
            ]
        )

    def save_plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping loss plot.")
            return
        df = pd.read_csv(self.csv_path)
        for col in ("val_loss", "val_metric"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        val = df[df["val_loss"].notna()]

        fig, (ax_loss, ax_m) = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss.plot(df["epoch"], df["train_loss"], label="train loss (per batch)")
        if not val.empty:
            ax_loss.plot(val["epoch"], val["val_loss"], linestyle="--",
                         label="val loss (whole set)")
            ax_m.plot(val["epoch"], val["val_metric"], color="tab:green",
                      label=f"val {self.metric_name}")
            if self.metric_name == "c-index":
                ax_m.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            ax_m.legend()
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_m.set_ylabel(self.metric_name)
        for ax in (ax_loss, ax_m):
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")


class StainMonitor:
    """Per-stain importance on the validation set, every epoch.

    For models with an exact per-stain split of the prediction
    (``model.decompose``, i.e. FlexMoEFusion), accumulated during the validation
    pass that already runs, over the biopsies that have each stain:

        n                 biopsies with the stain
        mean_abs_contrib  mean |contribution| to the prediction (model output units;
                          grows with the risk-score scale as training goes on)
        contrib_share     mean share of the biopsy's total |contribution| (over its
                          stains) carried by this stain: scale-free, so comparable
                          across epochs
        att_lift          mean attention received (model ``stain_att``, renormalised
                          over the biopsy's stains) x number of stains: 1 = the
                          stain gets its fair share

    Writes ``stain_monitor.csv`` (one row per epoch and stain) and redraws
    ``stain_monitor.png`` after every epoch, so a running job can be followed.
    """

    COLUMNS = ["epoch", "stain", "n", "mean_abs_contrib", "contrib_share", "att_lift"]

    def __init__(self, log_dir, stain_names, resume_epoch=None):
        self.stain_names = list(stain_names)
        self.csv_path = os.path.join(log_dir, "stain_monitor.csv")
        self.plot_path = os.path.join(log_dir, "stain_monitor.png")
        os.makedirs(log_dir, exist_ok=True)
        old = None
        if resume_epoch is not None and os.path.isfile(self.csv_path):
            old = pd.read_csv(self.csv_path)
            old = old[old["epoch"] <= resume_epoch] if list(old.columns) == self.COLUMNS else None
        (old if old is not None else pd.DataFrame(columns=self.COLUMNS)).to_csv(
            self.csv_path, index=False
        )
        self._reset()

    def _reset(self):
        S = len(self.stain_names)
        self.n = np.zeros(S)
        self.abs_sum = np.zeros(S)
        self.share_sum = np.zeros(S)
        self.lift_sum = np.zeros(S)

    def update(self, model, t, stain_att, mask):
        """One validation batch: ``t`` the per-stain tokens entering the head,
        ``stain_att`` and ``mask`` (B, S) from the same forward pass."""
        mask = mask.bool()
        contrib, _ = model.decompose(t)
        a = contrib.abs() * mask
        share = a / a.sum(1, keepdim=True).clamp_min(1e-12)
        lift = torch.softmax(stain_att.masked_fill(~mask, -torch.inf), dim=1)
        lift = lift * mask.sum(1, keepdim=True)
        self.n += mask.sum(0).cpu().numpy()
        self.abs_sum += a.sum(0).float().cpu().numpy()
        self.share_sum += (share * mask).sum(0).float().cpu().numpy()
        self.lift_sum += (lift * mask).sum(0).float().cpu().numpy()

    def end_epoch(self, epoch):
        """Write and print the epoch's statistics, redraw the plot, reset."""
        n = np.maximum(self.n, 1)
        rows = pd.DataFrame({
            "epoch": epoch,
            "stain": self.stain_names,
            "n": self.n.astype(int),
            "mean_abs_contrib": np.where(self.n > 0, self.abs_sum / n, np.nan),
            "contrib_share": np.where(self.n > 0, self.share_sum / n, np.nan),
            "att_lift": np.where(self.n > 0, self.lift_sum / n, np.nan),
        })
        rows.to_csv(self.csv_path, mode="a", header=False, index=False)
        seen = rows[rows["n"] > 0].sort_values("contrib_share", ascending=False)
        print("[stains] contribution share (attention lift): " + " | ".join(
            f"{r.stain} {r.contrib_share:.2f} ({r.att_lift:.2f})" for r in seen.itertuples()
        ))
        self._reset()
        try:
            self.save_plot()
        except Exception as e:  # a plotting failure must not stop training
            print(f"[WARN] stain monitor plot failed: {e}")

    def save_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return
        epochs = sorted(df["epoch"].unique())
        grids = {
            col: df.pivot(index="stain", columns="epoch", values=col)
            .reindex(index=self.stain_names, columns=epochs)
            for col in ("contrib_share", "att_lift")
        }
        # Attention lift is a polarity around 1 (fair share): a diverging scale
        # centred there, symmetric so equal distances read as equal colour.
        dev = np.nanmax(np.abs(grids["att_lift"].to_numpy() - 1.0)) if grids["att_lift"].notna().any().any() else 0.0
        half = max(float(dev), 0.05)
        panels = [
            ("contrib_share", "Share of |contribution| (over the biopsy's stains)",
             "Blues", Normalize(0, max(float(np.nanmax(grids["contrib_share"].to_numpy())), 1e-6))),
            ("att_lift", "Attention lift (1 = fair share)",
             "RdBu_r", Normalize(1 - half, 1 + half)),
        ]
        S = len(self.stain_names)
        fig, axes = plt.subplots(
            2, 1, figsize=(max(6.0, 0.18 * len(epochs) + 3), 2 * (0.28 * S + 1.2)),
            sharex=True,
        )
        for ax, (col, title, cmap, norm) in zip(axes, panels):
            im = ax.imshow(grids[col].to_numpy(dtype=float), aspect="auto", cmap=cmap,
                           norm=norm, interpolation="nearest")
            ax.set_yticks(range(S), self.stain_names, fontsize=8)
            ax.set_title(title, fontsize=10, loc="left")
            for side in ax.spines.values():
                side.set_visible(False)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        step = max(1, len(epochs) // 10)
        axes[-1].set_xticks(range(0, len(epochs), step), epochs[::step], fontsize=8)
        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


# Arguments that define the weight shapes (--stains sets the number of stain
# encoders): resuming with any of them changed is refused.
_ARCH_ARGS = (
    "task", "stains", "d_model", "stain_layers", "agg_layers", "n_heads",
    "pool_att_dim", "no_gated_pool", "share_stain_encoder", "per_slide",
    "fusion", "early_layers", "n_tokens", "n_experts", "top_k", "min_combo_count",
)


class WarmupSortedBatchSampler:
    """Flex-MoE's curriculum: for the first ``warm_up_epochs`` epochs, batches in a
    fixed order with the biopsies that have the most stains first (so the
    full-panel biopsies train every expert first); afterwards, the batches of
    ``base`` (shuffled or event-stratified). Call ``set_epoch`` (1-based) before
    each epoch."""

    def __init__(self, base, n_stains_per_sample, batch_size, warm_up_epochs, drop_last):
        self.base = base
        order = sorted(range(len(n_stains_per_sample)), key=lambda i: -n_stains_per_sample[i])
        self.sorted_batches = [
            order[i:i + batch_size] for i in range(0, len(order), batch_size)
        ]
        if drop_last and len(self.sorted_batches) > 1 and len(self.sorted_batches[-1]) < batch_size:
            self.sorted_batches.pop()
        self.warm_up_epochs = warm_up_epochs
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _warm(self):
        return self.epoch <= self.warm_up_epochs

    def __iter__(self):
        return iter(self.sorted_batches) if self._warm() else iter(self.base)

    def __len__(self):
        return len(self.sorted_batches) if self._warm() else len(self.base)


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def _predict(model, batch, device, return_att=False):
    # The datasets hand X over as float16 (half the transfer); compute in float32.
    slide_mask = batch.get("slide_mask")
    return model(
        batch["X"].to(device, non_blocking=True).float(),
        batch["stain_mask"].to(device, non_blocking=True),
        slide_mask=None if slide_mask is None else slide_mask.to(device, non_blocking=True),
        **({"return_att": True} if return_att else {}),
    )


def _seed_worker(worker_id):
    """Give each DataLoader worker its own numpy stream.

    The dataset draws patch subsets and stain-dropout masks with ``np.random``, but
    PyTorch only reseeds ``torch`` per worker — numpy is left alone, so every worker
    would otherwise replay the identical sequence of draws.  ``torch.initial_seed()``
    is already ``base_seed + worker_id`` with a fresh base seed per epoch, so keying
    off it varies the stream across both workers and epochs.
    """
    np.random.seed(torch.initial_seed() % 2**32)
    # With persistent workers this runs once per worker, not per epoch; each worker
    # then keeps drawing from its own stream, so epochs still differ.


def _available_ram():
    """MemAvailable of /proc/meminfo in bytes (free RAM + reclaimable cache), or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _cap_workers(num_workers, batch_bytes, share=1, prefetch=2):
    """Lower num_workers to what /dev/shm, the RAM and the CPU cores can hold.

    Each worker keeps ``prefetch`` finished batches queued in /dev/shm, and while
    it builds the next one holds its 32 samples plus their collated copy: about
    ``prefetch + 2`` batches of RAM. A batch is 32 x 14 stains x 512 patches x D
    float16 (1.2 GB for Virchow2's D=2560, 0.7 GB for UNI-v2), so 14 workers want
    ~65 GB per run. /dev/shm is a size limit, not reserved RAM, so its check alone
    lets two side-by-side runs fill RAM and swap, and the GPUs then wait on
    swapped-out batches. Limits, each divided by ``share`` (the runs side by
    side, which all see the same free memory at startup):
      - /dev/shm: 80% of its free space, for the queued batches;
      - RAM: 60% of MemAvailable, the rest left to the page cache that serves the
        feature reads;
      - CPU: one core per worker, one left for the training process.
    No-op where /proc and /dev/shm do not exist (Windows).
    """
    limits = {}
    try:
        free = shutil.disk_usage("/dev/shm").free
        limits[f"/dev/shm ({free / 1e9:.0f} GB free)"] = int(
            0.8 * free / share // (batch_bytes * prefetch)
        )
    except OSError:
        pass
    ram = _available_ram()
    if ram is not None:
        limits[f"RAM ({ram / 1e9:.0f} GB available)"] = int(
            0.6 * ram / share // (batch_bytes * (prefetch + 2))
        )
    if os.name == "posix":
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        limits[f"CPU ({cores} cores)"] = max(1, (cores or 1) // share - 1)
    if not limits:
        return num_workers
    why, fit = min(limits.items(), key=lambda kv: kv[1])
    if fit >= num_workers:
        return num_workers
    print(
        f"[WARN] --num_workers lowered from {num_workers} to {fit}, set by the {why} "
        f"shared by {share} run(s); one batch is {batch_bytes / 1e9:.1f} GB. "
        f"Lower --batch_size or --patches_per_stain to load with more workers."
    )
    return fit


def train_epoch(model, loader, optimizer, device, task, criterion, accumulation_steps=1):
    """Return (mean loss, seconds waiting for batches, seconds in total).

    The wait is the time spent in the loader before each batch; loss.item() syncs
    the GPU every step, so the rest of the epoch is the GPU's work. A high wait
    share means the GPU is starved by data loading. The running speed is printed
    10 times per epoch, so a long epoch shows its speed within minutes.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    start = tick = time.perf_counter()
    wait = 0.0
    total_aux = 0.0
    report = max(1, len(loader) // 10)
    for i, batch in enumerate(loader):
        if i and i % report == 0:
            elapsed = tick - start
            print(f"[time]   batch {i}/{len(loader)}: {i / elapsed:.2f} batches/s, "
                  f"{100 * wait / elapsed:.0f}% waiting for data")
        wait += time.perf_counter() - tick
        Y = batch["Y"].to(device, non_blocking=True)
        pred = _predict(model, batch, device)
        if task == "survival":
            loss = cox_ph_loss(pred, Y[:, 0], Y[:, 1])
        else:
            loss = criterion(pred, Y[:, 0])
        # Flex-MoE: the weighted router loss of the forward pass just run. The
        # logged train loss stays the task loss, comparable across fusion models.
        aux = getattr(model, "aux_loss", None)
        total = loss if aux is None else loss + aux
        (total / accumulation_steps).backward()
        total_loss += loss.item()
        if aux is not None:
            total_aux += aux.item()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        tick = time.perf_counter()
    if total_aux:
        print(f"[moe]  mean weighted router loss {total_aux / max(len(loader), 1):.4f}")
    return total_loss / max(len(loader), 1), wait, time.perf_counter() - start


@torch.no_grad()
def val_epoch(model, loader, device, task, criterion, monitor=None):
    """Return (loss, metric, preds, targets). With a ``StainMonitor``, also feed it
    every batch (the caller then calls ``monitor.end_epoch``).

    ``metric`` is the C-index (survival) or the MAE (regression, in whatever units
    the labels were fed in — see the normalisation block in main()).  ``preds`` and
    ``targets`` are returned so the caller can build the residual plot without a
    second pass over the validation set.

    The survival loss is the Cox loss over the whole validation set as one risk
    set, as in MIL.py: averaged over batches, most small batches hold zero or one
    event and the average is mostly noise. Regression keeps the per-batch mean,
    which does not depend on the batch composition.
    """
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    captured, hook = {}, None
    if monitor is not None:
        # The per-stain tokens entering the head, for the contribution split.
        hook = model.stain_pool.register_forward_pre_hook(
            lambda _mod, args: captured.__setitem__("t", args[0])
        )
    try:
        for batch in loader:
            Y = batch["Y"].to(device)
            if monitor is not None:
                pred, stain_att, _ = _predict(model, batch, device, return_att=True)
                monitor.update(model, captured["t"], stain_att,
                               batch["stain_mask"].to(device))
            else:
                pred = _predict(model, batch, device)
            if task != "survival":
                total_loss += criterion(pred, Y[:, 0]).item()
            preds.append(pred.float().cpu())
            targets.append(Y.float().cpu())
    finally:
        if hook is not None:
            hook.remove()

    preds_t, targets_t = torch.cat(preds), torch.cat(targets)
    if task == "survival":
        loss = cox_ph_loss(preds_t, targets_t[:, 0], targets_t[:, 1]).item()
    else:
        loss = total_loss / max(len(loader), 1)
    preds, targets = preds_t.numpy(), targets_t.numpy()
    if task == "survival":
        metric = concordance_index(preds, targets[:, 0], targets[:, 1])
    else:
        metric = float(np.abs(preds - targets[:, 0]).mean())
    return loss, metric, preds, targets


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------
def save_residual_plot(y_true, y_pred, path, label_name="label"):
    """Two-panel regression diagnostic, in raw label units.

    Left: predicted vs observed against the identity line — shows the regression-to-
    the-mean flattening that MIL models on small cohorts tend to produce.
    Right: residual vs predicted — a spread that widens with the prediction means the
    errors are multiplicative, which is the signal to model log(label) instead.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping residual plot.")
        return

    resid = y_pred - y_true
    mae = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid**2).mean()))
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax1.scatter(y_true, y_pred, s=20, alpha=0.6, edgecolor="none")
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="identity")
    ax1.set_xlabel(f"Observed {label_name}")
    ax1.set_ylabel(f"Predicted {label_name}")
    ax1.set_title(f"MAE {mae:.1f} | RMSE {rmse:.1f} | R² {r2:.2f}")
    ax1.legend()

    ax2.scatter(y_pred, resid, s=20, alpha=0.6, edgecolor="none")
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.axhline(
        resid.mean(), color="tab:red", ls=":", lw=1, label=f"bias {resid.mean():+.1f}"
    )
    ax2.set_xlabel(f"Predicted {label_name}")
    ax2.set_ylabel("Residual (pred − obs)")
    ax2.set_title("Residuals")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Residual plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Logs are redirected to files; without line buffering a line can sit in the
    # buffer for a long time (until it fills or the run ends).
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Train the multi-stain fusion model (UNICORN-style) on biopsies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="survival",
        choices=["survival", "regression"],
        help="survival: Cox partial likelihood on [time, event] labels. "
             "regression: MSE/MAE on a scalar label.",
    )

    # --- Data ----------------------------------------------------------------
    parser.add_argument(
        "--features_path",
        required=True,
        help="Flat feature folder ({features_path}/{slide}.h5).",
    )
    parser.add_argument("--labels_path", required=True, help="Flat .npy label folder.")
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="define_labels.py / define_regression_labels.py CSV: the biopsy, stain "
             "and train/val split of every slide.",
    )
    parser.add_argument(
        "--stains",
        nargs="+",
        default=["PAS", "PASM", "HES|HE", "Congo", "AFOG"],
        help="Stain vocabulary, in model order. Use '|' to merge source stain names "
             "onto one model stain (e.g. 'HES|HE'). Names are matched "
             "case-insensitively, ignoring punctuation.",
    )
    parser.add_argument(
        "--min_stains",
        type=int,
        default=1,
        help="Drop biopsies with fewer than this many of the selected stains.",
    )
    parser.add_argument("--file_ext", default=".h5", choices=[".h5", ".npy"])
    parser.add_argument(
        "--authorized_slides_csv",
        default=None,
        help="CSV of slide names to keep; all other slides are ignored.",
    )
    parser.add_argument(
        "--patches_per_stain",
        type=int,
        default=512,
        help="Patches sampled per stain per biopsy — per slide with --per_slide. "
             "Drives memory: one sample costs n_stains x patches_per_stain x feat_dim "
             "floats (x max_slides with --per_slide).",
    )
    parser.add_argument(
        "--per_slide",
        action="store_true",
        help="Encode each slide of a stain separately and let an attention pool "
             "weight the slides into the stain token, instead of pooling all the "
             "stain's patches into one bag. --patches_per_stain then counts patches "
             "per slide.",
    )
    parser.add_argument(
        "--max_slides",
        type=int,
        default=None,
        help="With --per_slide (default 4): slide slots per stain. A stain with more slides "
             "uses a random subset each training epoch and the first ones (by name) "
             "for validation.",
    )
    parser.add_argument(
        "--stain_dropout",
        type=float,
        default=0.0,
        help="Training augmentation: probability of masking out each available stain, "
             "drawn independently per stain per epoch. Teaches the model to predict "
             "from partial stain sets instead of leaning on the most available stain. "
             "At least one stain always survives. 0 disables; try 0.2-0.3.",
    )

    # --- Model ---------------------------------------------------------------
    parser.add_argument(
        "--fusion",
        default="unicorn",
        choices=["unicorn", "early", "flexmoe"],
        help="unicorn: per-stain encoders -> one token per stain -> transformer "
             "aggregator. early: one shared projection + stain embedding for every "
             "patch, then a single attention pool over the patches of all stains "
             "(--stain_layers, --agg_layers and --share_stain_encoder do not apply). "
             "flexmoe: Flex-MoE intermediate fusion (multistain_model.FlexMoEFusion): "
             "--n_tokens tokens per stain, missing stains filled from a learned bank, "
             "--agg_layers fusion layers with a sparse mixture of experts in every "
             "other one.",
    )
    flex = parser.add_argument_group("Flex-MoE (--fusion flexmoe)")
    flex.add_argument("--n_tokens", type=int, default=16, help="Tokens per stain.")
    flex.add_argument(
        "--min_combo_count",
        type=int,
        default=10,
        help="Stain combinations with at least this many training biopsies get their "
             "own missing-stain embeddings and specialist expert; rarer ones share one.",
    )
    flex.add_argument(
        "--n_experts",
        type=int,
        default=None,
        help="Experts per MoE layer (default: one per combination, the full panel and "
             "the shared rare one included).",
    )
    flex.add_argument("--top_k", type=int, default=4, help="Experts per token.")
    flex.add_argument(
        "--gate_loss_weight", type=float, default=0.01, help="Weight of the router loss."
    )
    flex.add_argument(
        "--warm_up_epochs",
        type=int,
        default=5,
        help="Epochs in a fixed order, biopsies with the most stains first, before "
             "shuffled (or event-stratified) batches.",
    )
    parser.add_argument(
        "--early_layers",
        type=int,
        default=0,
        help="--fusion early: transformer layers over the joint bag of all stains' "
             "patches before pooling (0 = attention pooling only). Cost grows with "
             "the square of n_stains x patches_per_stain (x max_slides).",
    )
    parser.add_argument("--d_model", type=int, default=256, help="Token dimension.")
    parser.add_argument(
        "--stain_layers", type=int, default=2, help="Layers per stain encoder."
    )
    parser.add_argument(
        "--agg_layers", type=int, default=2, help="Layers in the stain aggregator."
    )
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool_att_dim", type=int, default=128)
    parser.add_argument(
        "--no_gated_pool",
        action="store_true",
        help="Use plain instead of gated attention pooling.",
    )
    parser.add_argument(
        "--share_stain_encoder",
        action="store_true",
        help="Share one patch encoder across all stains instead of one per stain "
             "(more sample-efficient when the rarer stains have few biopsies).",
    )

    # --- Training ------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "mae"],
        help="Regression loss (ignored when --task survival).",
    )
    parser.add_argument(
        "--label_name",
        default="eGFR",
        help="Display name of the regression target, used on the residual plot axes.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--shm_share",
        type=int,
        default=1,
        help="Number of runs training side by side on this machine; --num_workers "
             "is capped to 1/shm_share of the free /dev/shm, RAM and CPU cores.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints_multistain")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--save_every", type=int, default=10, help="Checkpoint every N epochs (0 = final only)."
    )
    parser.add_argument(
        "--event_sampling",
        action="store_true",
        help="Event-stratified batches (survival only): every training batch gets the "
             "same share of events and a spread of follow-up times, as in MIL.py.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a <checkpoint_name>_last.pth (written every epoch): restores "
             "the weights, optimizer, epoch and best val metric, continues up to "
             "--epochs, and appends to the existing loss log. Pass the same "
             "architecture arguments as the original run.",
    )

    args = parser.parse_args()
    if args.early_layers and args.fusion != "early":
        parser.error("--early_layers only applies with --fusion early.")
    if args.max_slides is not None and not args.per_slide:
        parser.error("--max_slides only applies with --per_slide.")
    if args.per_slide and args.max_slides is None:
        args.max_slides = 4
    if args.event_sampling and args.task != "survival":
        parser.error("--event_sampling only applies to --task survival.")
    if args.fusion == "flexmoe" and args.per_slide:
        parser.error("--fusion flexmoe does not support --per_slide.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  task: {args.task}")

    # --- Index biopsies -------------------------------------------------------
    vocab = StainVocabulary(args.stains)
    print(f"Stain vocabulary ({len(vocab)}): {vocab.names}")

    table = load_labels_table(args.labels_csv)
    records = index_biopsies(
        args.features_path,
        args.labels_path,
        table,
        vocab,
        file_ext=args.file_ext,
        authorized_slides=load_authorized_slides(args.authorized_slides_csv),
        min_stains=args.min_stains,
    )
    if not records:
        parser.error(
            f"No biopsies found — check the feature/label paths and --stains "
            f"(looked for {args.file_ext} files; use --file_ext .npy for a .npy folder)."
        )

    expected_label_len = 2 if args.task == "survival" else 1
    if len(records[0]["label"]) < expected_label_len:
        parser.error(
            f"--task {args.task} expects labels with at least {expected_label_len} "
            f"value(s), but the .npy files hold {len(records[0]['label'])}. "
            "Point --labels_path at the right label folder."
        )

    # Test biopsies are held out entirely; evaluate them at the end with
    # analyze_stain_weights.py --split test.
    train_records, val_records, test_records = split_records(records, table)
    print(
        f"Biopsies — train: {len(train_records)} | val: {len(val_records)}"
        f" | test (held out): {len(test_records)}"
    )

    # --- Label normalisation (regression only) --------------------------------
    # The head is LayerNorm -> Linear with default init, so it starts predicting ~0
    # while raw eGFR sits around 60: the first epochs are spent walking the head bias
    # up to the mean, against a weight decay that pulls it back to 0.  Z-scoring puts
    # the target on the scale the head already predicts on, and makes the loss
    # magnitude (and hence the shared --lr / --weight_decay defaults) comparable with
    # the scale-free Cox objective.
    #
    # Statistics come from the *training* biopsies only — including val would leak,
    # and one value per biopsy rather than per slide keeps multi-slide biopsies from
    # getting extra weight.
    label_mean, label_std = 0.0, 1.0
    if args.task == "regression":
        y = np.array([r["label"][0] for r in train_records], dtype=np.float64)
        label_mean, label_std = float(y.mean()), float(y.std())
        if label_std == 0:
            parser.error("Training labels have zero variance — nothing to regress on.")
        for rec in train_records + val_records:
            rec["label"] = (rec["label"] - label_mean) / label_std
        print(
            f"Train {args.label_name}: {label_mean:.1f} ± {label_std:.1f} "
            f"(range {y.min():.1f}–{y.max():.1f}) — labels z-scored"
        )

    train_ds = MultiStainBiopsyDataset(
        train_records,
        n_stains=len(vocab),
        patches_per_stain=args.patches_per_stain,
        random_subsample=True,
        stain_dropout=args.stain_dropout,
        per_slide=args.per_slide,
        max_slides=args.max_slides,
        dtype=torch.float16,
    )
    # Validation keeps every stain (and the deterministic patch selection) so the
    # metric measures the model, not the draw.
    val_ds = (
        MultiStainBiopsyDataset(
            val_records,
            n_stains=len(vocab),
            patches_per_stain=args.patches_per_stain,
            random_subsample=False,
            feat_dim=train_ds.feat_dim,
            per_slide=args.per_slide,
            max_slides=args.max_slides,
            dtype=torch.float16,
        )
        if val_records
        else None
    )

    if args.event_sampling:
        # Labels are raw [time, event] for survival (only regression is z-scored).
        batch_sampler = EventStratifiedBatchSampler.from_labels(
            [r["label"][1] for r in train_records],
            [r["label"][0] for r in train_records],
            args.batch_size,
        )
        lo, hi = batch_sampler.events_per_batch()
        print(
            f"Event-stratified batches: {len(batch_sampler)} per epoch, "
            f"{lo}-{hi} events each"
        )
        loader_kwargs = dict(batch_sampler=batch_sampler)
    else:
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=True,
            # A short last batch makes a tiny Cox risk set; stratified batches
            # are all near batch_size, so they need no dropping.
            drop_last=(args.task == "survival" and len(train_ds) > args.batch_size),
        )
    warmup_sampler = None
    if args.fusion == "flexmoe" and args.warm_up_epochs > 0:
        drop_last = args.task == "survival" and len(train_ds) > args.batch_size
        base = loader_kwargs.get("batch_sampler") or torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_ds), args.batch_size, drop_last
        )
        warmup_sampler = WarmupSortedBatchSampler(
            base, [len(r["slides"]) for r in train_records], args.batch_size,
            args.warm_up_epochs, drop_last,
        )
        loader_kwargs = dict(batch_sampler=warmup_sampler)
        print(f"Flex-MoE warm-up: {args.warm_up_epochs} epoch(s) of batches sorted by "
              "stain count, most stains first")
    # Workers stay alive across epochs (starting them costs seconds per worker,
    # twice per epoch) and batches land in pinned memory for an asynchronous copy
    # to the GPU.
    num_workers = _cap_workers(
        args.num_workers,
        batch_bytes=args.batch_size * len(vocab) * (args.max_slides if args.per_slide else 1)
        * args.patches_per_stain * train_ds.feat_dim * 2,  # float16
        share=args.shm_share,
    )
    worker_kwargs = dict(
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        worker_init_fn=_seed_worker,
        **worker_kwargs,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **worker_kwargs,
        )
        if val_ds is not None
        else None
    )

    # --- Model ---------------------------------------------------------------
    combos = None
    if args.fusion == "flexmoe":
        combos = build_combos(
            [r["slides"].keys() for r in train_records], len(vocab), args.min_combo_count
        )
        known = {frozenset(c) for c in combos}
        covered = sum(frozenset(r["slides"]) in known for r in train_records)
        n_full = sum(len(r["slides"]) == len(vocab) for r in train_records)
        print(
            f"Flex-MoE: {len(combos)} stain combinations with >= {args.min_combo_count} "
            f"training biopsies (full panel: {n_full} biopsies) cover {covered}/"
            f"{len(train_records)}; the other {len(train_records) - covered} share one index"
        )
        if n_full == 0:
            print("[WARN] no training biopsy has every stain: the full-panel expert and "
                  "the G-Router balancing get no training signal.")
        model = FlexMoEFusion(
            in_dim=train_ds.feat_dim,
            n_stains=len(vocab),
            combos=combos,
            d_model=args.d_model,
            n_tokens=args.n_tokens,
            stain_layers=args.stain_layers,
            fusion_layers=args.agg_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            n_experts=args.n_experts,
            top_k=args.top_k,
            gate_loss_weight=args.gate_loss_weight,
            share_stain_encoder=args.share_stain_encoder,
            out_dim=1,
        ).to(device)
    elif args.fusion == "early":
        model = EarlyFusionMIL(
            in_dim=train_ds.feat_dim,
            n_stains=len(vocab),
            d_model=args.d_model,
            n_layers=args.early_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            pool_att_dim=args.pool_att_dim,
            gated=not args.no_gated_pool,
            out_dim=1,
        ).to(device)
    else:
        model = MultiStainFusion(
            in_dim=train_ds.feat_dim,
            n_stains=len(vocab),
            d_model=args.d_model,
            stain_layers=args.stain_layers,
            agg_layers=args.agg_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            pool_att_dim=args.pool_att_dim,
            gated=not args.no_gated_pool,
            out_dim=1,
            share_stain_encoder=args.share_stain_encoder,
            per_slide=args.per_slide,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {args.fusion} fusion, feat_dim={train_ds.feat_dim}, "
        f"{n_params / 1e6:.1f}M parameters"
    )

    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"multistain_{args.task}.pth"
    last_path = os.path.join(
        args.checkpoint_dir, f"{os.path.splitext(checkpoint_name)[0]}_last.pth"
    )
    best_metric, best_epoch, start_epoch = None, None, 1
    if args.resume:
        done, best_metric, best_epoch = load_last_checkpoint(
            args.resume, model, optimizer, args, device, parser, _ARCH_ARGS
        )
        start_epoch = done + 1
        best_info = (
            f", best val metric {best_metric:.4f} at epoch {best_epoch}"
            if best_epoch is not None else ""
        )
        print(f"Resumed from {args.resume}: {done} epochs done{best_info}.")
        if start_epoch > args.epochs:
            print(f"Nothing to do: --epochs {args.epochs} already reached.")

    log_dir = args.log_dir or args.checkpoint_dir
    metric_name = "c-index" if args.task == "survival" else f"MAE({args.label_name})"
    logger = MultiStainLogger(
        log_dir, metric_name, resume_epoch=start_epoch - 1 if args.resume else None
    )
    print(f"Loss log: {logger.csv_path}  |  val metric: {metric_name}")
    monitor = None
    if hasattr(model, "decompose") and val_loader is not None:
        monitor = StainMonitor(
            log_dir, vocab.names, resume_epoch=start_epoch - 1 if args.resume else None
        )
        print(f"Stain monitor: {monitor.csv_path}")

    with open(os.path.join(args.checkpoint_dir, "multistain_config.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "stains": args.stains,
                "stain_names": vocab.names,
                "feat_dim": train_ds.feat_dim,
                "patches_per_stain": args.patches_per_stain,
                "stain_dropout": args.stain_dropout,
                "d_model": args.d_model,
                "stain_layers": args.stain_layers,
                "agg_layers": args.agg_layers,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
                "pool_att_dim": args.pool_att_dim,
                "gated": not args.no_gated_pool,
                "share_stain_encoder": args.share_stain_encoder,
                "fusion": args.fusion,
                "early_layers": args.early_layers,
                "per_slide": args.per_slide,
                "max_slides": args.max_slides,
                # Flex-MoE only (None otherwise): the combination vocabulary fixes the
                # shape of the missing-stain bank, so inference needs it verbatim.
                "combos": combos,
                "n_tokens": args.n_tokens,
                "n_experts": args.n_experts,
                "top_k": args.top_k,
                "gate_loss_weight": args.gate_loss_weight,
                "warm_up_epochs": args.warm_up_epochs,
                # Inverse transform for the model output: raw = pred * std + mean.
                # Identity (0, 1) for survival, where the Cox score has no units.
                "label_name": args.label_name,
                "label_mean": label_mean,
                "label_std": label_std,
            },
            f,
            indent=2,
        )

    # --- Training -------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        if warmup_sampler is not None:
            warmup_sampler.set_epoch(epoch)
        train_loss, wait_s, train_s = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.task,
            criterion,
            accumulation_steps=args.accumulation_steps,
        )
        timing = (
            f"[time] epoch {epoch}: train {train_s:.0f} s "
            f"({len(train_loader) / max(train_s, 1e-9):.2f} batches/s, "
            f"{100 * wait_s / max(train_s, 1e-9):.0f}% waiting for data)"
        )

        if val_loader is not None:
            val_start = time.perf_counter()
            val_loss, metric, _, _ = val_epoch(
                model, val_loader, device, args.task, criterion, monitor=monitor
            )
            if monitor is not None:
                monitor.end_epoch(epoch)
            timing += f", val {time.perf_counter() - val_start:.0f} s"
        print(timing)
        if val_loader is not None:
            # The MAE comes back in z-score units; report it in raw label units so the
            # log reads in mL/min/1.73m². Monotone, so best-model selection is unaffected.
            if args.task == "regression":
                metric *= label_std
            logger.log(epoch, train_loss, val_loss, metric)
            print(
                f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}  {metric_name}={metric:.4f}"
            )
            # C-index: higher is better; MAE: lower is better.
            improved = best_metric is None or (
                metric > best_metric if args.task == "survival" else metric < best_metric
            )
            if improved:
                best_metric, best_epoch = metric, epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_dir, f"multistain_{args.task}_best.pth"),
                )
        else:
            logger.log(epoch, train_loss)
            print(f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"multistain_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt}")

        save_last_checkpoint(
            last_path, model, optimizer, epoch, best_metric, best_epoch, args
        )

    final_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    if best_metric is not None:
        print(f"Best val {metric_name}: {best_metric:.4f} at epoch {best_epoch}")

    logger.save_plot()

    # --- Residual diagnostics on the best checkpoint ---------------------------
    if args.task == "regression" and best_metric is not None:
        best_path = os.path.join(
            args.checkpoint_dir, f"multistain_{args.task}_best.pth"
        )
        model.load_state_dict(torch.load(best_path, map_location=device))
        _, _, preds, targets = val_epoch(
            model, val_loader, device, args.task, criterion
        )
        save_residual_plot(
            targets[:, 0] * label_std + label_mean,
            preds * label_std + label_mean,
            os.path.join(log_dir, "residuals.png"),
            label_name=args.label_name,
        )


if __name__ == "__main__":
    main()
