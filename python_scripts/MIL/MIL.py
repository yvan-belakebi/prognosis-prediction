"""
MIL.py — Train and validate ABMIL or DeepGraphSurv for WSI prognosis prediction.

Two-phase training:
  Phase 1 (pretrain): train on non-IgA data for --pretrain_epochs epochs (no validation).
  Phase 2 (finetune): train on combined IgA + IgA_registry for remaining epochs,
                      validated on the combined IgA + IgA_registry val sets.

Usage (ABMIL, two-phase with stain filtering on IgA): (on server)
    python MIL.py --model_type abmil \\
        --pretrain_features_path WSI/non_IgA/UNI2-h_feats \\
        --pretrain_labels_path   WSI/non_IgA/labels \\
        --pretrain_epochs 20 \\
        --features_paths WSI/IgA/UNI2-h_feats WSI/IgA_registry/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels        WSI/IgA_registry/labels \\
        --val_csv followup_data/survival_validation_files.csv \\
        --stain_filter PAS \\
        --stain_csvs followup_data/labels_combined.csv none \\
        --epochs 50

Usage (DeepGraphSurv, two-phase — coords auto-read from features .h5, no --coords_paths needed):
    python python_scripts/MIL/MIL.py --model_type deepgraphsurv \\
        --pretrain_features_path WSI/non_IgA/UNI2-h_feats \\
        --pretrain_labels_path   WSI/non_IgA/labels \\
        --pretrain_epochs 10 \\
        --features_paths WSI/IgA/UNI2-h_feats WSI/IgA_registry/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels        WSI/IgA_registry/labels \\
        --val_csv followup_data/survival_validation_files.csv \\
        --stain_filter PAS \\
        --stain_csvs followup_data/labels_combined.csv followup_data/labels_combined.csv \\
        --checkpoint_dir checkpoints_10_epochs_pretraining/ \\
        --log_dir results/losses_10_pretraining/ \\
        --dropout 0.1 --save_every 5 --batch_size 4


Usage (single-repo, no pretraining):
    python MIL.py --model_type abmil \\
        --features_paths WSI/IgA/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels \\
        --val_csv followup_data/survival_validation_files.csv

Labels (.npy, shape (2,)): [time_to_first_event, censoring_indicator]
"""

import csv
import os
import sys
import argparse
from functools import partial

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
from torchmil.models import deepgraphsurv as dgs_module
from torchmil.models import patch_gcn as patch_gcn_module

from mil_utils import (
    load_val_names,
    make_collate_fn,
    build_dataset,
    BiopsySampler,
)


# ---------------------------------------------------------------------------
# Survival loss
# ---------------------------------------------------------------------------
def cox_ph_loss(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """Breslow-approximation of the Cox partial likelihood."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -(risk - log_cumsum) * event
    return loss.sum() / event.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Model forward helpers
# ---------------------------------------------------------------------------
def _forward(model, batch, model_type: str) -> torch.Tensor:
    if model_type == "abmil":
        return model(batch["X"], batch["mask"])
    # deepgraphsurv: adj may be sparse (COO) and/or float64 from numpy; normalise both.
    adj = batch["adj"]
    if adj.is_sparse:
        adj = adj.to_dense()
    adj = adj.float()
    return model(batch["X"], adj, batch["mask"])


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def train_epoch(
    model, loader, optimizer, device, model_type: str, accumulation_steps: int = 1
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        risk = _forward(model, batch, model_type)
        loss = cox_ph_loss(risk, batch["Y"][:, 0].float(), batch["Y"][:, 1].float())
        # Scale so that accumulated gradients match a single full-batch step.
        # Note: for Cox PH the risk set is per micro-batch, which is approximate
        # but acceptable in practice.
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / len(loader)


def val_epoch(model, loader, device, model_type: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            risk = _forward(model, batch, model_type)
            loss = cox_ph_loss(risk, batch["Y"][:, 0].float(), batch["Y"][:, 1].float())
            total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Loss logger
# ---------------------------------------------------------------------------
class LossLogger:
    """Writes loss values to a CSV and saves a loss-curve plot at the end."""

    COLUMNS = ["epoch", "phase", "train_loss", "val_loss"]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "loss_log.csv")
        self.plot_path = os.path.join(log_dir, "loss_curves.png")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)

    def log(
        self, epoch: int, phase: str, train_loss: float, val_loss: float | None = None
    ):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    phase,
                    f"{train_loss:.6f}",
                    "" if val_loss is None else f"{val_loss:.6f}",
                ]
            )

    def save_plot(self, pretrain_epochs: int = 0):
        if not _HAS_MATPLOTLIB:
            print("matplotlib not available — skipping loss plot.")
            return
        df = pd.read_csv(self.csv_path)
        df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["epoch"], df["train_loss"], label="train loss")
        val_rows = df["val_loss"].notna()
        if val_rows.any():
            ax.plot(
                df.loc[val_rows, "epoch"],
                df.loc[val_rows, "val_loss"],
                linestyle="--",
                label="val loss",
            )
        if pretrain_epochs > 0:
            ax.axvspan(
                0.5, pretrain_epochs + 0.5, alpha=0.08, color="gray", label="pretrain"
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train ABMIL or DeepGraphSurv for WSI prognosis."
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "deepgraphsurv", "patchgcn"],
        help="Model architecture (default: abmil).",
    )

    # --- Phase 1: pretrain on non-IgA (no val files) -------------------------
    parser.add_argument(
        "--pretrain_features_path",
        default=None,
        help="Non-IgA features folder for pretraining.",
    )
    parser.add_argument(
        "--pretrain_labels_path",
        default=None,
        help="Non-IgA labels folder for pretraining.",
    )
    parser.add_argument(
        "--pretrain_coords_path",
        default=None,
        help="Non-IgA coords folder for pretraining (deepgraphsurv / patchgcn only).",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Epochs to train on non-IgA before switching to IgA datasets (default: 0).",
    )

    # --- Phase 2: finetune on IgA + IgA_registry -----------------------------
    parser.add_argument(
        "--features_paths",
        nargs="+",
        required=True,
        help="Training features folder(s) (one per repo, e.g. IgA and IgA_registry).",
    )
    parser.add_argument(
        "--labels_paths",
        nargs="+",
        required=True,
        help="Training labels folder(s).",
    )
    parser.add_argument(
        "--coords_paths",
        nargs="+",
        default=None,
        help="Training coords folder(s) (deepgraphsurv / patchgcn only). "
             "When --file_ext .h5 and omitted, defaults to --features_paths "
             "(coords are read from the same .h5 file as features).",
    )
    parser.add_argument(
        "--file_ext",
        default=".h5",
        choices=[".h5", ".npy"],
        help="File extension for feature and coordinate bags (default: .h5). "
             "Use .npy for the legacy pipeline.",
    )
    parser.add_argument(
        "--label_ext",
        default=".h5",
        choices=[".h5", ".npy"],
        help="File extension for label bags (default: .h5). "
             "Use .npy for the legacy pipeline.",
    )
    parser.add_argument(
        "--val_csv",
        default=None,
        help=(
            "CSV listing validation slide basenames ('file_name' column or headerless). "
            "The same feature/label directories are used; bags are split at load time."
        ),
    )

    # --- Stain filtering (e.g. for IgA_light which contains multiple stains) -
    parser.add_argument(
        "--stain_filter",
        default=None,
        help="Keep only bags with this value in the 'Stain' column (e.g. PAS).",
    )
    parser.add_argument(
        "--stain_csvs",
        nargs="+",
        default=None,
        help=(
            "Path to labels_combined.csv for each entry in --features_paths, "
            "in the same order. Use 'none' for repos that need no filtering."
        ),
    )

    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", help="Folder to save checkpoints."
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        help="Final checkpoint filename (default: <model_type>_model.pth).",
    )
    parser.add_argument(
        "--log_dir",
        default="results/losses",
        help="Folder for loss_log.csv and loss_curves.png (default: same as --checkpoint_dir).",
    )

    # --- Hyperparameters -----------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total training epochs including pretrain (default: 50).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "Gradient accumulation steps. Use with a reduced --batch_size to lower "
            "peak GPU memory while preserving the effective batch size "
            "(e.g. --batch_size 4 --accumulation_steps 4 ≈ batch_size 16). "
            "Default: 1 (no accumulation)."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--att_dim", type=int, default=128, help="Attention dimension.")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dim for DeepGraphSurv / PatchGCN (default: feat_dim).",
    )
    # DeepGraphSurv-specific
    parser.add_argument(
        "--n_layers_rep", type=int, default=1, help="DeepGraphSurv rep GCN layers."
    )
    parser.add_argument(
        "--n_layers_att", type=int, default=1, help="DeepGraphSurv att GCN layers."
    )
    parser.add_argument(
        "--K", type=int, default=5, help="Chebyshev polynomial order (deepgraphsurv)."
    )
    # PatchGCN-specific
    parser.add_argument(
        "--n_gcn_layers", type=int, default=4, help="Number of GCN layers (patchgcn)."
    )
    parser.add_argument(
        "--mlp_depth", type=int, default=1, help="MLP depth after GCN (patchgcn)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (deepgraphsurv / patchgcn).",
    )
    parser.add_argument(
        "--dist_thr",
        type=float,
        default=1.5,
        help="Adjacency distance threshold (deepgraphsurv).",
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=None,
        help=(
            "Randomly subsample each slide to at most this many patches before "
            "the adjacency matrix is built. Required for PatchGCN / DeepGraphSurv "
            "on large slides (e.g. --max_patches 4096)."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 = only final).",
    )
    parser.add_argument(
        "--biopsy_sampling",
        action="store_true",
        help=(
            "Sample one slide per biopsy per epoch. "
            "Removes repeated survival labels from the Cox risk set for biopsies "
            "with multiple slides."
        ),
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

    args = parser.parse_args()
    torch.cuda.empty_cache()

    # --- Validate argument combinations --------------------------------------
    n_train = len(args.features_paths)
    if len(args.labels_paths) != n_train:
        parser.error(
            "--features_paths and --labels_paths must have the same number of entries."
        )

    do_pretrain = args.pretrain_features_path is not None
    if do_pretrain:
        if args.pretrain_labels_path is None:
            parser.error(
                "--pretrain_labels_path is required when --pretrain_features_path is set."
            )
        if args.pretrain_epochs <= 0:
            parser.error(
                "--pretrain_epochs must be > 0 when --pretrain_features_path is set."
            )
    if args.pretrain_epochs > 0 and not do_pretrain:
        parser.error("--pretrain_features_path is required when --pretrain_epochs > 0.")

    if args.model_type in ("deepgraphsurv", "patchgcn"):
        if args.coords_paths is None:
            if args.file_ext == ".h5":
                # Coords are embedded in the features .h5; reuse the same directories.
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
                parser.error(
                    f"--pretrain_coords_path is required for {args.model_type} pretraining."
                )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bag_keys = (
        ["X", "Y", "adj", "coords"]
        if args.model_type in ("deepgraphsurv", "patchgcn")
        else ["X", "Y"]
    )

    def _none_list(paths, n):
        return paths if paths is not None else [None] * n

    # --- Build datasets -------------------------------------------------------
    val_names = load_val_names(args.val_csv)
    train_dataset, val_dataset, _, _ = build_dataset(
        args.features_paths,
        args.labels_paths,
        _none_list(args.coords_paths, n_train),
        bag_keys,
        args.dist_thr,
        val_names=val_names,
        stain_csvs=args.stain_csvs,
        stain_filter=args.stain_filter,
        max_biopsies=args.max_biopsies,
        file_ext=args.file_ext,
        label_ext=args.label_ext,
    )

    pretrain_dataset = None
    if do_pretrain:
        pretrain_dataset, _, _, _ = build_dataset(
            [args.pretrain_features_path],
            [args.pretrain_labels_path],
            [args.pretrain_coords_path] if args.pretrain_coords_path is not None else [None],
            bag_keys,
            args.dist_thr,
            val_names=None,
            stain_csvs=[None],
            stain_filter=None,
            file_ext=args.file_ext,
            label_ext=args.label_ext,
        )

    val_info = f" | Val bags: {len(val_dataset)}" if val_dataset is not None else ""
    print(f"Finetune train bags: {len(train_dataset)}{val_info}")
    if pretrain_dataset is not None:
        print(f"Pretrain bags: {len(pretrain_dataset)}")

    # Subsample patches before adj is built to avoid OOM on large slides.
    _collate = make_collate_fn(
        partial(
            collate_fn, sparse=(args.model_type not in ("deepgraphsurv", "patchgcn"))
        ),
        max_patches=args.max_patches,
    )

    _train_sampler = BiopsySampler(train_dataset) if args.biopsy_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(_train_sampler is None),
        sampler=_train_sampler,
        collate_fn=_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
        )
        if val_dataset is not None
        else None
    )
    if pretrain_dataset is not None:
        _pretrain_sampler = (
            BiopsySampler(pretrain_dataset) if args.biopsy_sampling else None
        )
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=(_pretrain_sampler is None),
            sampler=_pretrain_sampler,
            collate_fn=_collate,
        )
    else:
        pretrain_loader = None

    # --- Model ---------------------------------------------------------------
    if args.model_type == "abmil":
        model = abmil_module.ABMIL(att_dim=args.att_dim)
    else:
        # Resolve feat_dim from the first available sample across datasets.
        ref_dataset = (
            pretrain_dataset if pretrain_dataset is not None else train_dataset
        )
        feat_dim = int(ref_dataset[0]["X"].shape[-1])
        if args.model_type == "deepgraphsurv":
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
        else:  # patchgcn
            model = patch_gcn_module.PatchGCN(
                in_shape=(feat_dim,),
                n_gcn_layers=args.n_gcn_layers,
                mlp_depth=args.mlp_depth,
                hidden_dim=args.hidden_dim,
                att_dim=args.att_dim,
                dropout=args.dropout,
            )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"{args.model_type}_model.pth"

    log_dir = args.log_dir or args.checkpoint_dir
    logger = LossLogger(log_dir)
    print(f"Loss log: {logger.csv_path}")

    # --- Phase 1: pretrain on non-IgA (no validation) ------------------------
    if do_pretrain:
        print(f"\n--- Pretrain phase: {args.pretrain_epochs} epochs on non-IgA ---")
        for epoch in range(1, args.pretrain_epochs + 1):
            train_loss = train_epoch(
                model,
                pretrain_loader,
                optimizer,
                device,
                args.model_type,
                accumulation_steps=args.accumulation_steps,
            )
            logger.log(epoch, "pretrain", train_loss)
            print(
                f"[Pretrain {epoch:>3d}/{args.pretrain_epochs}] train_loss={train_loss:.4f}"
            )

            if args.save_every > 0 and epoch % args.save_every == 0:
                ckpt = os.path.join(
                    args.checkpoint_dir, f"{args.model_type}_pretrain_epoch{epoch}.pth"
                )
                torch.save(model.state_dict(), ckpt)
                print(f"  → checkpoint: {ckpt}")

    # --- Phase 2: finetune on IgA + IgA_registry -----------------------------
    print(f"\n--- Finetune phase: {finetune_epochs} epochs on IgA datasets ---")
    for epoch in range(1, finetune_epochs + 1):
        global_epoch = epoch + args.pretrain_epochs
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.model_type,
            accumulation_steps=args.accumulation_steps,
        )

        if val_loader is not None:
            val_loss = val_epoch(model, val_loader, device, args.model_type)
            logger.log(global_epoch, "finetune", train_loss, val_loss)
            print(
                f"[Epoch {global_epoch:>3d}/{args.epochs}]"
                f" train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )
        else:
            logger.log(global_epoch, "finetune", train_loss)
            print(
                f"[Epoch {global_epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}"
            )

        if args.save_every > 0 and global_epoch % args.save_every == 0:
            ckpt = os.path.join(
                args.checkpoint_dir, f"{args.model_type}_epoch{global_epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt}")

    final_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

    logger.save_plot(pretrain_epochs=args.pretrain_epochs)


if __name__ == "__main__":
    main()
