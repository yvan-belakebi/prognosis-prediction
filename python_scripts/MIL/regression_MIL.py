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

Pretraining (--pretrain_features_path) can be validated too: pass
--pretrain_val_csv with a non-IgA validation list (define_regression_labels.py
--val_non_iga).  It is a separate list from --val_csv because the two phases
train on different cohorts; both losses land in the same loss_log.csv, tagged
by the phase column.

Usage:
    python regression_MIL.py --model_type abmil \\
        --features_paths WSI/registry_IgA/UNI2-h_feats \\
        --labels_paths   WSI/registry_IgA/labels_regression \\
        --val_csv validation_files_csvs/regression_validation_files.csv \\
        --epochs 50

Usage (validated pretraining on non-IgA):
    python regression_MIL.py --model_type abmil \\
        --pretrain_features_path WSI/non_IgA/UNI2-h_feats \\
        --pretrain_labels_path   WSI/non_IgA/labels_regression \\
        --pretrain_val_csv validation_files_csvs/regression_validation_files_non_IgA.csv \\
        --pretrain_epochs 10 \\
        --features_paths WSI/IgA_registry/UNI2-h_feats \\
        --labels_paths   WSI/IgA_registry/labels_regression \\
        --val_csv validation_files_csvs/regression_validation_files.csv \\
        --epochs 50
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
    discover_bags,
    load_val_names,
    load_authorized_slides,
    make_collate_fn,
    build_dataset,
    drop_empty_bags,
    BagDataset,
    BiopsySampler,
    LossLogger as _BaseLossLogger,
    _val_segments,
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
        adj = batch["adj"]
        if adj.is_sparse:
            adj = adj.to_dense()
        return model(batch["X"], adj.float(), batch["mask"])
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

    COLUMNS = ["epoch", "phase", "train_loss", "val_loss", "val_mae", "val_rmse"]

    def log(self, epoch, phase, train_loss, val_loss=None, val_mae=None, val_rmse=None):
        self._append(
            [
                epoch,
                phase,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_mae is None else f"{val_mae:.4f}",
                "" if val_rmse is None else f"{val_rmse:.4f}",
            ]
        )

    def save_plot(self, pretrain_epochs=0):
        if not _HAS_MATPLOTLIB:
            return
        df = pd.read_csv(self.csv_path)
        for col in ("val_loss", "val_mae", "val_rmse"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss, ax_mae = axes
        for ax in axes:
            if pretrain_epochs > 0:
                ax.axvspan(
                    0.5,
                    pretrain_epochs + 0.5,
                    alpha=0.08,
                    color="gray",
                    label="pretrain",
                )
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)

        ax_loss.plot(df["epoch"], df["train_loss"], label="train loss")
        for grp, label in _val_segments(df, "val_loss"):
            ax_loss.plot(grp["epoch"], grp["val_loss"], linestyle="--", label=label)
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        for grp, label in _val_segments(df, "val_mae"):
            ax_mae.plot(
                grp["epoch"], grp["val_mae"], color="tab:orange", label=label
            )
        ax_mae.set_ylabel("MAE")
        ax_mae.legend()

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

    # Pretrain
    parser.add_argument("--pretrain_features_path", default=None)
    parser.add_argument("--pretrain_labels_path", default=None)
    parser.add_argument("--pretrain_coords_path", default=None)
    parser.add_argument(
        "--pretrain_val_csv",
        default=None,
        help="Validation slide list for the pretrain phase (non-IgA), e.g. "
        "validation_files_csvs/regression_validation_files_non_IgA.csv written by "
        "define_regression_labels.py --val_non_iga. Those bags are held out of "
        "the pretrain training set and scored each pretrain epoch. Omitted (the "
        "default) means pretraining runs unvalidated on every non-IgA bag.",
    )
    parser.add_argument("--pretrain_epochs", type=int, default=0)

    # Data
    parser.add_argument("--features_paths", nargs="+", required=True)
    parser.add_argument("--labels_paths", nargs="+", required=True)
    parser.add_argument(
        "--coords_paths",
        nargs="+",
        default=None,
        help="Coord folder(s) for patchgcn. When --file_ext .h5 and omitted, "
             "defaults to --features_paths.",
    )
    parser.add_argument(
        "--file_ext",
        default=".h5",
        choices=[".h5", ".npy"],
        help="File extension for feature and coordinate bags (default: .h5). "
             "Use .npy for the legacy pipeline.",
    )
    parser.add_argument("--val_csv", default=None)
    parser.add_argument(
        "--authorized_slides_csv",
        default=None,
        help=(
            "CSV listing authorized slide basenames ('file_name' column or headerless). "
            "When set, only bags whose file basename appears in this list are loaded "
            "into the train/val dataloaders (applies to --features_paths, not pretrain)."
        ),
    )
    parser.add_argument("--stain_filter", default=None)
    parser.add_argument("--stain_csvs", nargs="+", default=None)

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
        help=(
            "Maximum number of training biopsies across all datasets combined. "
            "The budget is split equally across datasets; if one has fewer biopsies "
            "than its share, all of them are kept and the surplus is redistributed "
            "to the others. Has no effect on the validation set."
        ),
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

    # Validate
    n_train = len(args.features_paths)
    if len(args.labels_paths) != n_train:
        parser.error("--features_paths and --labels_paths must have the same length.")

    do_pretrain = args.pretrain_features_path is not None
    if do_pretrain and args.pretrain_labels_path is None:
        parser.error(
            "--pretrain_labels_path is required with --pretrain_features_path."
        )
    if do_pretrain and args.pretrain_epochs <= 0:
        parser.error("--pretrain_epochs must be > 0 with --pretrain_features_path.")
    if args.pretrain_val_csv is not None and not do_pretrain:
        parser.error(
            "--pretrain_val_csv is only meaningful with --pretrain_features_path."
        )

    is_graph = args.model_type in _GRAPH_MODELS
    bag_keys = ["X", "Y", "adj", "coords"] if is_graph else ["X", "Y"]
    if is_graph:
        if args.coords_paths is None:
            if args.file_ext == ".h5":
                args.coords_paths = args.features_paths
            else:
                parser.error(f"--coords_paths is required for {args.model_type}.")
        if len(args.coords_paths) != n_train:
            parser.error(
                "--coords_paths must have the same number of entries as --features_paths."
            )
        if do_pretrain and args.pretrain_coords_path is None:
            if args.file_ext == ".h5":
                args.pretrain_coords_path = args.pretrain_features_path
            else:
                parser.error(f"--pretrain_coords_path is required for {args.model_type} pretraining.")

    if args.stain_csvs is not None:
        if args.stain_filter is None:
            parser.error("--stain_filter is required when --stain_csvs is set.")
        if len(args.stain_csvs) != n_train:
            parser.error(
                "--stain_csvs must have the same number of entries as --features_paths."
            )

    finetune_epochs = args.epochs - args.pretrain_epochs
    if finetune_epochs <= 0:
        parser.error("--pretrain_epochs must be less than --epochs.")

    def _none_list(paths, n):
        return paths if paths is not None else [None] * n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets
    val_names = load_val_names(args.val_csv)
    authorized_slides = load_authorized_slides(args.authorized_slides_csv)
    train_dataset, val_dataset, train_labels, val_labels = build_dataset(
        args.features_paths,
        args.labels_paths,
        _none_list(args.coords_paths, n_train),
        bag_keys,
        args.dist_thr,
        val_names=val_names,
        stain_csvs=args.stain_csvs,
        stain_filter=args.stain_filter,
        scan_labels_fn=scan_labels,
        max_biopsies=args.max_biopsies,
        file_ext=args.file_ext,
        authorized_slides=authorized_slides,
    )

    print(
        f"Train bags: {len(train_dataset)}"
        + (f" | Val bags: {len(val_dataset)}" if val_dataset else "")
    )
    if train_labels is not None and len(train_labels):
        print(
            f"Train {args.label_name}: {train_labels.mean():.1f} ± {train_labels.std():.1f}"
        )

    pretrain_dataset = pretrain_val_dataset = None
    if do_pretrain:
        pretrain_names = discover_bags(args.pretrain_features_path)
        pretrain_labelled = set(discover_bags(args.pretrain_labels_path, extensions=(".npy",)))
        pretrain_names = [n for n in pretrain_names if n in pretrain_labelled]
        pretrain_names = drop_empty_bags(
            args.pretrain_features_path, pretrain_names, args.file_ext
        )

        # The pretrain val list is separate from --val_csv: it names non-IgA
        # bags, while --val_csv names IgA + registry bags for the finetune
        # phase.  Without --pretrain_val_csv every bag stays in training, which
        # is the previous behaviour.
        pretrain_val_names = load_val_names(args.pretrain_val_csv)

        def _make_pretrain_ds(names):
            return BagDataset(
                features_path=args.pretrain_features_path,
                labels_path=args.pretrain_labels_path,
                coords_path=args.pretrain_coords_path,
                bag_keys=bag_keys,
                dist_thr=args.dist_thr,
                bag_names=names,
                file_ext=args.file_ext,
                label_ext=".npy",
            )

        if pretrain_val_names is not None:
            held_out = [n for n in pretrain_names if n in pretrain_val_names]
            pretrain_names = [n for n in pretrain_names if n not in pretrain_val_names]
            if held_out:
                pretrain_val_dataset = _make_pretrain_ds(held_out)
            else:
                print(
                    f"Warning: no pretrain bag matched {args.pretrain_val_csv} "
                    f"— pretraining will run unvalidated."
                )

        pretrain_dataset = _make_pretrain_ds(pretrain_names)
        pretrain_val_info = (
            f" | Pretrain val bags: {len(pretrain_val_dataset)}"
            if pretrain_val_dataset is not None
            else ""
        )
        print(f"Pretrain bags: {len(pretrain_dataset)}{pretrain_val_info}")

    _collate = make_collate_fn(
        partial(collate_fn, sparse=not is_graph),
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
    pretrain_val_loader = (
        _make_loader(pretrain_val_dataset, False) if pretrain_val_dataset else None
    )
    if args.biopsy_sampling:
        pretrain_loader = (
            _make_loader(pretrain_dataset, True, BiopsySampler(pretrain_dataset))
            if pretrain_dataset
            else None
        )
        train_loader = _make_loader(train_dataset, True, BiopsySampler(train_dataset))
    else:
        pretrain_loader = _make_loader(pretrain_dataset, True) if pretrain_dataset else None
        train_loader = _make_loader(train_dataset, True)

    # Model
    ref = pretrain_dataset if pretrain_dataset is not None else train_dataset
    feat_dim = int(ref[0]["X"].shape[-1])
    model = build_model(args.model_type, feat_dim, args).to(device)

    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"{args.model_type}_regression.pth"
    log_dir = args.log_dir or args.checkpoint_dir
    logger = LossLogger(log_dir)
    print(f"Loss: {args.loss.upper()}  |  Log: {logger.csv_path}")

    # Phase 1: pretrain
    if do_pretrain:
        print(f"\n--- Pretrain: {args.pretrain_epochs} epochs ---")
        for epoch in range(1, args.pretrain_epochs + 1):
            tl = train_epoch(
                model,
                pretrain_loader,
                optimizer,
                criterion,
                device,
                args.model_type,
                args.accumulation_steps,
            )
            if pretrain_val_loader is not None:
                vl, mae, rmse = val_epoch(
                    model, pretrain_val_loader, criterion, device, args.model_type
                )
                logger.log(epoch, "pretrain", tl, vl, mae, rmse)
                print(
                    f"[Pretrain {epoch:>3d}/{args.pretrain_epochs}] loss={tl:.4f}"
                    f"  val_loss={vl:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}"
                )
            else:
                logger.log(epoch, "pretrain", tl)
                print(f"[Pretrain {epoch:>3d}/{args.pretrain_epochs}] loss={tl:.4f}")
            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.checkpoint_dir,
                        f"{args.model_type}_pretrain_epoch{epoch}.pth",
                    ),
                )

    # Phase 2: finetune
    print(f"\n--- Finetune: {finetune_epochs} epochs ---")
    for epoch in range(1, finetune_epochs + 1):
        ge = epoch + args.pretrain_epochs
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
            logger.log(ge, "finetune", tl, vl, mae, rmse)
            print(
                f"[{ge:>3d}/{args.epochs}] loss={tl:.4f}  val_loss={vl:.4f}"
                f"  MAE={mae:.2f}  RMSE={rmse:.2f}"
            )
        else:
            logger.log(ge, "finetune", tl)
            print(f"[{ge:>3d}/{args.epochs}] loss={tl:.4f}")

        if args.save_every > 0 and ge % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch{ge}.pth"),
            )

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name))
    print(f"Model saved to {os.path.join(args.checkpoint_dir, checkpoint_name)}")

    logger.save_plot(pretrain_epochs=args.pretrain_epochs)

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
