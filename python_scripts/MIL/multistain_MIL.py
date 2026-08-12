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
        --features_paths WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested \\
        --labels_paths   WSI/IgA/trident/labels \\
        --stain_csvs     label_csvs/labels_unfiltered.csv \\
        --stains PAS PASM "HES|HE" Congo AFOG \\
        --val_csv validation_files_csvs/survival_validation_20pct_both.csv \\
        --patches_per_stain 512 --batch_size 8 --epochs 50 \\
        --checkpoint_dir checkpoints_multistain --log_dir results/losses_multistain

Usage (regression, eGFR):
    python python_scripts/MIL/multistain_MIL.py --task regression \\
        --features_paths WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested \\
        --labels_paths   WSI/IgA/trident/labels_regression \\
        --stain_csvs     label_csvs/labels_unfiltered.csv \\
        --stains PAS PASM "HES|HE" Congo AFOG \\
        --val_csv validation_files_csvs/regression_validation_files.csv

A stain argument may list aliases separated by '|' ("HES|HE"); the first name is the
one reported.  Stains absent from --stains are ignored, and biopsies keep whichever
of the selected stains they happen to have — missing stains are masked out.

Alongside the checkpoints, ``multistain_config.json`` records the stain vocabulary and
the architecture, so evaluation and inference can rebuild the exact same model.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
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

from mil_utils import load_val_names, load_authorized_slides, LossLogger  # noqa: E402
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
    """Loss log with one extra metric column (C-index for survival, MAE for regression)."""

    COLUMNS = ["epoch", "phase", "train_loss", "val_loss", "val_metric"]

    def log(self, epoch, phase, train_loss, val_loss=None, val_metric=None):
        self._append(
            [
                epoch,
                phase,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_metric is None else f"{val_metric:.4f}",
            ]
        )


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def _predict(model, batch, device):
    return model(batch["X"].to(device), batch["stain_mask"].to(device))


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
    """Return (avg_loss, metric) — metric is the C-index (survival) or the MAE."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for batch in loader:
        Y = batch["Y"].to(device)
        pred = _predict(model, batch, device)
        if task == "survival":
            loss = cox_ph_loss(pred, Y[:, 0], Y[:, 1])
        else:
            loss = criterion(pred, Y[:, 0])
        total_loss += loss.item()
        preds.append(pred.cpu())
        targets.append(Y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if task == "survival":
        metric = concordance_index(preds, targets[:, 0], targets[:, 1])
    else:
        metric = float(np.abs(preds - targets[:, 0]).mean())
    return total_loss / max(len(loader), 1), metric


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
        "--features_paths",
        nargs="+",
        required=True,
        help="Biopsy-nested feature folder(s), one per cohort "
             "({features_path}/{biopsy}/{slide}.h5).",
    )
    parser.add_argument(
        "--labels_paths",
        nargs="+",
        required=True,
        help="Label folder(s), same order and count as --features_paths.",
    )
    parser.add_argument(
        "--stain_csvs",
        nargs="+",
        required=True,
        help="define_labels.py CSV(s) with biopsy_number / file_name / stain columns, "
             "same order as --features_paths. Pass one CSV to reuse it for all.",
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
        "--val_csv",
        default=None,
        help="CSV of validation slide (or biopsy) names — a biopsy is held out when "
             "any of its slides is listed.",
    )
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints_multistain")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--save_every", type=int, default=10, help="Checkpoint every N epochs (0 = final only)."
    )

    args = parser.parse_args()

    n_cohorts = len(args.features_paths)
    if len(args.labels_paths) != n_cohorts:
        parser.error("--features_paths and --labels_paths must have the same length.")
    if len(args.stain_csvs) == 1:
        args.stain_csvs = args.stain_csvs * n_cohorts
    if len(args.stain_csvs) != n_cohorts:
        parser.error("--stain_csvs must have 1 entry or one per --features_paths.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  task: {args.task}")

    # --- Index biopsies -------------------------------------------------------
    vocab = StainVocabulary(args.stains)
    print(f"Stain vocabulary ({len(vocab)}): {vocab.names}")

    authorized = load_authorized_slides(args.authorized_slides_csv)
    records = []
    for fp, lp, sc in zip(args.features_paths, args.labels_paths, args.stain_csvs):
        records += index_biopsies(
            fp,
            lp,
            sc,
            vocab,
            file_ext=args.file_ext,
            authorized_slides=authorized,
            min_stains=args.min_stains,
        )
    if not records:
        parser.error("No biopsies found — check the feature/label paths and --stains.")

    expected_label_len = 2 if args.task == "survival" else 1
    if len(records[0]["label"]) < expected_label_len:
        parser.error(
            f"--task {args.task} expects labels with at least {expected_label_len} "
            f"value(s), but the .npy files hold {len(records[0]['label'])}. "
            "Point --labels_paths at the right label folder."
        )

    train_records, val_records = split_records(records, load_val_names(args.val_csv))
    print(f"Biopsies — train: {len(train_records)} | val: {len(val_records)}")

    train_ds = MultiStainBiopsyDataset(
        train_records,
        n_stains=len(vocab),
        patches_per_stain=args.patches_per_stain,
        random_subsample=True,
    )
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=(args.task == "survival" and len(train_ds) > args.batch_size),
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
    log_dir = args.log_dir or args.checkpoint_dir
    logger = MultiStainLogger(log_dir)
    metric_name = "c-index" if args.task == "survival" else "MAE"
    print(f"Loss log: {logger.csv_path}  |  val metric: {metric_name}")

    with open(os.path.join(args.checkpoint_dir, "multistain_config.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "stains": args.stains,
                "stain_names": vocab.names,
                "feat_dim": train_ds.feat_dim,
                "patches_per_stain": args.patches_per_stain,
                "d_model": args.d_model,
                "stain_layers": args.stain_layers,
                "agg_layers": args.agg_layers,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
                "pool_att_dim": args.pool_att_dim,
                "gated": not args.no_gated_pool,
                "share_stain_encoder": args.share_stain_encoder,
            },
            f,
            indent=2,
        )

    # --- Training -------------------------------------------------------------
    best_metric = None
    for epoch in range(1, args.epochs + 1):
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
            val_loss, metric = val_epoch(model, val_loader, device, args.task, criterion)
            logger.log(epoch, "train", train_loss, val_loss, metric)
            print(
                f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}  {metric_name}={metric:.4f}"
            )
            # C-index: higher is better; MAE: lower is better.
            improved = best_metric is None or (
                metric > best_metric if args.task == "survival" else metric < best_metric
            )
            if improved:
                best_metric = metric
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoint_dir, f"multistain_{args.task}_best.pth"),
                )
        else:
            logger.log(epoch, "train", train_loss)
            print(f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"multistain_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt}")

    final_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    if best_metric is not None:
        print(f"Best val {metric_name}: {best_metric:.4f}")

    logger.save_plot()


if __name__ == "__main__":
    main()
