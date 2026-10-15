"""
regression_MIL.py — Train and validate MIL models for WSI regression (e.g. eGFR).

Labels: .npy files containing a single float64 scalar (produced by
define_regression_labels.py).

Supported models:
    abmil     — Attention-Based MIL (Ilse et al., 2018)
    dsmil     — Dual-Stream MIL (Li et al., 2021)
    transmil  — Transformer MIL (Shao et al., 2021)
    patchgcn  — Patch-based Graph CNN (Chen et al., 2021); coords auto-read from features .h5

Training loss: MSE by default; switch to MAE with --loss mae.

Data: one flat features folder (--features_path), the flat .npy labels from
define_regression_labels.py (--labels_path) and its labels CSV (--labels_csv),
which supplies each slide's biopsy, stain (for --stains) and train/val split.

Usage:
    python python_scripts/MIL/regression_MIL.py --model_type abmil \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels_regression \\
        --labels_csv    label_csvs/labels_regression.csv \\
        --stains PAS --epochs 50
"""

import os
import sys
import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

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

from torchmil.data import collate_fn
from torchmil.models import abmil as abmil_module
from torchmil.models import dsmil as dsmil_module
from torchmil.models import transmil as transmil_module
from torchmil.models import patch_gcn as patch_gcn_module

from mil_utils import (
    load_authorized_slides,
    load_labels_table,
    make_collate_fn,
    build_dataset,
    BiopsySampler,
    LossLogger as _BaseLossLogger,
)

_GRAPH_MODELS = {"patchgcn"}


# ---------------------------------------------------------------------------
# Forward / label helpers
# ---------------------------------------------------------------------------


def _forward(model, batch, model_type: str) -> torch.Tensor:
    """Return predictions of shape (batch_size,)."""
    if model_type == "transmil":
        return model(batch["X"])
    if model_type == "patchgcn":
        # Sparse adjacency kept as is: a dense one is B x N x N (see MIL._forward).
        return model(batch["X"], batch["adj"].float(), batch["mask"])
    return model(batch["X"], batch["mask"])


def _labels(batch, device) -> torch.Tensor:
    """Return regression targets (batch_size,) as float32."""
    return batch["Y"].view(-1).float().to(device)


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------


def train_epoch(
    model, loader, optimizer, criterion, device, model_type, accumulation_steps=1
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        preds = _forward(model, batch, model_type)
        loss = criterion(preds, _labels(batch, device))
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / len(loader)


def val_epoch(
    model, loader, criterion, device, model_type
) -> tuple[float, float, float]:
    """Returns (avg_loss, MAE, RMSE)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = _forward(model, batch, model_type)
            lbls = _labels(batch, device)
            total_loss += criterion(preds, lbls).item()
            all_preds.append(preds.cpu())
            all_labels.append(lbls.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mae = (all_preds - all_labels).abs().mean().item()
    rmse = ((all_preds - all_labels) ** 2).mean().sqrt().item()
    return total_loss / len(loader), mae, rmse


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def scan_labels(labels_path: str, bag_names: list) -> np.ndarray:
    """Read float regression labels from .npy files."""
    return np.array(
        [float(np.load(os.path.join(labels_path, f"{n}.npy"))) for n in bag_names],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Loss logger
# ---------------------------------------------------------------------------


class LossLogger(_BaseLossLogger):
    """Regression logger: adds val_mae / val_rmse columns and an MAE plot panel."""

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_mae", "val_rmse"]

    def log(self, epoch, train_loss, val_loss=None, val_mae=None, val_rmse=None):
        self._append(
            [
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_mae is None else f"{val_mae:.4f}",
                "" if val_rmse is None else f"{val_rmse:.4f}",
            ]
        )

    def save_plot(self):
        if not _HAS_MATPLOTLIB:
            return
        df = pd.read_csv(self.csv_path)
        for col in ("val_loss", "val_mae", "val_rmse"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        val = df[df["val_loss"].notna()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss, ax_mae = axes
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)

        ax_loss.plot(df["epoch"], df["train_loss"], label="train loss")
        if not val.empty:
            ax_loss.plot(val["epoch"], val["val_loss"], linestyle="--", label="val loss")
            ax_mae.plot(val["epoch"], val["val_mae"], color="tab:orange", label="val mae")
            ax_mae.legend()
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_mae.set_ylabel("MAE")

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")


# ---------------------------------------------------------------------------
# Result scatter plot
# ---------------------------------------------------------------------------


def plot_regression_results(
    model, loader, device, model_type, log_dir, label_name="eGFR"
):
    """Scatter plot of predicted vs actual values on the validation set."""
    if not _HAS_MATPLOTLIB:
        print("matplotlib not available — skipping scatter plot.")
        return
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = _forward(model, batch, model_type).cpu().tolist()
            lbls = _labels(batch, device).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    mae = np.abs(preds - labels).mean()
    rmse = np.sqrt(((preds - labels) ** 2).mean())
    # R²
    ss_res = ((labels - preds) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.5, s=20, color="#2196F3", edgecolors="none")
    lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"Actual {label_name}")
    ax.set_ylabel(f"Predicted {label_name}")
    ax.set_title(f"Validation — Predicted vs Actual {label_name}")
    ax.text(
        0.05,
        0.95,
        f"MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²   = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(log_dir, "regression_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved to {path}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(model_type, feat_dim, args) -> nn.Module:
    if model_type == "abmil":
        model = abmil_module.ABMIL(
            in_shape=(feat_dim,), att_dim=args.att_dim, gated=args.gated
        )
    elif model_type == "dsmil":
        model = dsmil_module.DSMIL(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            nonlinear_q=args.nonlinear_q,
            nonlinear_v=args.nonlinear_v,
            dropout=args.dropout,
        )
    elif model_type == "transmil":
        model = transmil_module.TransMIL(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    else:  # patchgcn
        model = patch_gcn_module.PatchGCN(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            n_gcn_layers=args.n_gcn_layers,
            mlp_depth=args.mlp_depth,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    # Replace classifier head with single-output (regression)
    model.classifier = nn.LazyLinear(1)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train MIL models for WSI regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "dsmil", "transmil", "patchgcn"],
    )
    parser.add_argument(
        "--label_name",
        default="eGFR",
        help="Human-readable name of the regression target (for plots).",
    )

    # Loss
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "mae"],
        help="Training loss: MSE (default) or MAE.",
    )

    # --- Data ----------------------------------------------------------------
    parser.add_argument(
        "--features_path",
        required=True,
        help="Flat features folder (<slide>.h5 or .npy).",
    )
    parser.add_argument(
        "--labels_path",
        required=True,
        help="Flat .npy labels folder written by define_labels.py.",
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="Labels CSV written by define_labels.py: the biopsy, stain and "
             "train/val split of every slide.",
    )
    parser.add_argument(
        "--stains",
        nargs="+",
        default=None,
        help="Stains to train on (default: all). Use '|' to merge source stain "
             "names (e.g. 'HES|HE'); matching ignores case and punctuation.",
    )
    parser.add_argument(
        "--coords_path",
        default=None,
        help="Coords folder (deepgraphsurv / patchgcn only). When --file_ext .h5 "
             "and omitted, defaults to --features_path (coords are read from the "
             "same .h5 file as features).",
    )
    parser.add_argument(
        "--file_ext",
        default=".h5",
        choices=[".h5", ".npy"],
        help="File extension for feature and coordinate bags (default: .h5). "
             "Use .npy for the legacy pipeline.",
    )
    parser.add_argument(
        "--authorized_slides_csv",
        default=None,
        help=(
            "CSV listing authorized slide names ('file_name' column or headerless). "
            "When set, only those slides are loaded."
        ),
    )

    # Output
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--log_dir", default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument(
        "--biopsy_sampling",
        action="store_true",
        help="Sample one slide per biopsy per epoch to avoid inflated loss from repeated biopsies.",
    )
    parser.add_argument(
        "--max_biopsies",
        type=int,
        default=None,
        help="Maximum number of training biopsies (all slides of a selected biopsy are kept). Has no effect on the validation set.",
    )

    # Model architecture
    parser.add_argument("--att_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dist_thr", type=float, default=1.5)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--nonlinear_q", action="store_true")
    parser.add_argument("--nonlinear_v", action="store_true")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_gcn_layers", type=int, default=4)
    parser.add_argument("--mlp_depth", type=int, default=1)

    args = parser.parse_args()

    is_graph = args.model_type in _GRAPH_MODELS
    bag_keys = ["X", "Y", "adj", "coords"] if is_graph else ["X", "Y"]
    if is_graph and args.coords_path is None:
        if args.file_ext == ".h5":
            args.coords_path = args.features_path
        else:
            parser.error(f"--coords_path is required for {args.model_type}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets
    table = load_labels_table(args.labels_csv)
    train_dataset, val_dataset, train_labels, val_labels = build_dataset(
        args.features_path,
        args.labels_path,
        args.coords_path,
        bag_keys,
        args.dist_thr,
        table,
        stains=args.stains,
        scan_labels_fn=scan_labels,
        max_biopsies=args.max_biopsies,
        file_ext=args.file_ext,
        authorized_slides=load_authorized_slides(args.authorized_slides_csv),
    )

    print(
        f"Train bags: {len(train_dataset)}"
        + (f" | Val bags: {len(val_dataset)}" if val_dataset else "")
    )
    if train_labels is not None and len(train_labels):
        print(
            f"Train {args.label_name}: {train_labels.mean():.1f} ± {train_labels.std():.1f}"
        )

    _collate = make_collate_fn(
        partial(collate_fn, sparse=True),
        max_patches=args.max_patches,
    )

    def _make_loader(ds, shuffle, sampler=None):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            collate_fn=_collate,
        )

    val_loader = _make_loader(val_dataset, False) if val_dataset else None
    train_loader = _make_loader(
        train_dataset,
        True,
        BiopsySampler(train_dataset, table) if args.biopsy_sampling else None,
    )

    # Model
    feat_dim = int(train_dataset[0]["X"].shape[-1])
    model = build_model(args.model_type, feat_dim, args).to(device)

    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"{args.model_type}_regression.pth"
    log_dir = args.log_dir or args.checkpoint_dir
    logger = LossLogger(log_dir)
    print(f"Loss: {args.loss.upper()}  |  Log: {logger.csv_path}")

    for epoch in range(1, args.epochs + 1):
        tl = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.model_type,
            args.accumulation_steps,
        )
        if val_loader is not None:
            vl, mae, rmse = val_epoch(
                model, val_loader, criterion, device, args.model_type
            )
            logger.log(epoch, tl, vl, mae, rmse)
            print(
                f"[{epoch:>3d}/{args.epochs}] loss={tl:.4f}  val_loss={vl:.4f}"
                f"  MAE={mae:.2f}  RMSE={rmse:.2f}"
            )
        else:
            logger.log(epoch, tl)
            print(f"[{epoch:>3d}/{args.epochs}] loss={tl:.4f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch{epoch}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name))
    print(f"Model saved to {os.path.join(args.checkpoint_dir, checkpoint_name)}")

    logger.save_plot()

    if val_loader is not None:
        plot_regression_results(
            model,
            val_loader,
            device,
            args.model_type,
            log_dir,
            label_name=args.label_name,
        )


if __name__ == "__main__":
    main()
