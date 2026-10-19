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
import sys

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
from multistain_model import MultiStainFusion  # noqa: E402
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


# Arguments that define the weight shapes (--stains sets the number of stain
# encoders): resuming with any of them changed is refused.
_ARCH_ARGS = (
    "task", "stains", "d_model", "stain_layers", "agg_layers", "n_heads",
    "pool_att_dim", "no_gated_pool", "share_stain_encoder",
)


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def _predict(model, batch, device):
    return model(batch["X"].to(device), batch["stain_mask"].to(device))


def _seed_worker(worker_id):
    """Give each DataLoader worker its own numpy stream.

    The dataset draws patch subsets and stain-dropout masks with ``np.random``, but
    PyTorch only reseeds ``torch`` per worker — numpy is left alone, so every worker
    would otherwise replay the identical sequence of draws.  ``torch.initial_seed()``
    is already ``base_seed + worker_id`` with a fresh base seed per epoch, so keying
    off it varies the stream across both workers and epochs.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def train_epoch(model, loader, optimizer, device, task, criterion, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        Y = batch["Y"].to(device)
        pred = _predict(model, batch, device)
        if task == "survival":
            loss = cox_ph_loss(pred, Y[:, 0], Y[:, 1])
        else:
            loss = criterion(pred, Y[:, 0])
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, device, task, criterion):
    """Return (loss, metric, preds, targets).

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
    for batch in loader:
        Y = batch["Y"].to(device)
        pred = _predict(model, batch, device)
        if task != "survival":
            total_loss += criterion(pred, Y[:, 0]).item()
        preds.append(pred.float().cpu())
        targets.append(Y.float().cpu())

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
        help="Patches sampled per stain per biopsy. Drives memory: one sample costs "
             "n_stains x patches_per_stain x feat_dim floats.",
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
    if args.event_sampling and args.task != "survival":
        parser.error("--event_sampling only applies to --task survival.")

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
        parser.error("No biopsies found — check the feature/label paths and --stains.")

    expected_label_len = 2 if args.task == "survival" else 1
    if len(records[0]["label"]) < expected_label_len:
        parser.error(
            f"--task {args.task} expects labels with at least {expected_label_len} "
            f"value(s), but the .npy files hold {len(records[0]['label'])}. "
            "Point --labels_path at the right label folder."
        )

    train_records, val_records = split_records(records, table)
    print(f"Biopsies — train: {len(train_records)} | val: {len(val_records)}")

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
    train_loader = DataLoader(
        train_ds,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_ds is not None
        else None
    )

    # --- Model ---------------------------------------------------------------
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
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: feat_dim={train_ds.feat_dim}, {n_params / 1e6:.1f}M parameters")

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
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.task,
            criterion,
            accumulation_steps=args.accumulation_steps,
        )

        if val_loader is not None:
            val_loss, metric, _, _ = val_epoch(
                model, val_loader, device, args.task, criterion
            )
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
