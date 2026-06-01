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
from torch.utils.data import DataLoader, ConcatDataset

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

from torchmil.datasets import ProcessedMILDataset
from torchmil.data import collate_fn
from torchmil.models import abmil as abmil_module
from torchmil.models import deepgraphsurv as dgs_module
from torchmil.models import patch_gcn as patch_gcn_module


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
def train_epoch(model, loader, optimizer, device, model_type: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        risk = _forward(model, batch, model_type)
        loss = cox_ph_loss(risk, batch["Y"][:, 0].float(), batch["Y"][:, 1].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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
# Dataset helpers
# ---------------------------------------------------------------------------
def get_filtered_bag_names(features_path, stain_csv, stain_filter):
    """Return sorted bag names from features_path whose Stain matches stain_filter.

    Returns None when stain_csv is None or 'none', meaning no filtering is applied
    and ProcessedMILDataset will auto-discover all bags in features_path.
    """
    if stain_csv is None or stain_csv.lower() == "none":
        return None
    df = pd.read_csv(stain_csv)
    matching = set(df.loc[df["Stain"] == stain_filter, "file_name"].astype(str))
    available = {
        os.path.splitext(f)[0]
        for f in os.listdir(features_path)
        if f.endswith(".npy") or f.endswith(".h5")
    }
    return sorted(matching & available)


def load_val_names(val_csv):
    """Load slide basenames from a CSV into a set.

    Accepts a CSV with a 'file_name' column (header row) or a headerless
    single-column file (one basename per row). Returns None when val_csv is None.
    """
    if val_csv is None:
        return None
    raw = pd.read_csv(val_csv, header=None, dtype=str)
    col = raw.iloc[:, 0].str.strip()
    if col.iloc[0].lower() == "file_name":
        col = col.iloc[1:]
    return set(col)


def build_dataset(
    features_paths,
    labels_paths,
    coords_paths,
    bag_keys,
    dist_thr,
    val_names=None,
    stain_csvs=None,
    stain_filter=None,
):
    """Build train and (optionally) val datasets from lists of paths.

    When val_names is provided, each repository is split: bags whose basename
    appears in val_names go to val, the rest go to train.
    Returns (train_dataset, val_dataset); val_dataset is None when val_names is None.
    """
    stain_csvs = stain_csvs if stain_csvs is not None else [None] * len(features_paths)
    train_datasets, val_datasets = [], []

    for fp, lp, cp, sc in zip(features_paths, labels_paths, coords_paths, stain_csvs):
        filtered = get_filtered_bag_names(fp, sc, stain_filter)
        if filtered is None:
            available = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(fp)
                if f.endswith(".npy") or f.endswith(".h5")
            )
        else:
            available = filtered

        if val_names is not None:
            train_names = [n for n in available if n not in val_names]
            val_names_here = [n for n in available if n in val_names]
        else:
            train_names = available
            val_names_here = []

        train_datasets.append(
            ProcessedMILDataset(
                features_path=fp, labels_path=lp, coords_path=cp,
                bag_keys=bag_keys, dist_thr=dist_thr, bag_names=train_names,
            )
        )
        if val_names_here:
            val_datasets.append(
                ProcessedMILDataset(
                    features_path=fp, labels_path=lp, coords_path=cp,
                    bag_keys=bag_keys, dist_thr=dist_thr, bag_names=val_names_here,
                )
            )

    train_ds = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    if not val_datasets:
        return train_ds, None
    val_ds = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    return train_ds, val_ds


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
        help="Training coords folder(s) (deepgraphsurv / patchgcn only).",
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
        "--dropout", type=float, default=0.0, help="Dropout rate (deepgraphsurv / patchgcn)."
    )
    parser.add_argument(
        "--dist_thr",
        type=float,
        default=1.5,
        help="Adjacency distance threshold (deepgraphsurv).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 = only final).",
    )

    args = parser.parse_args()

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
            parser.error(f"--coords_paths is required for {args.model_type}.")
        if len(args.coords_paths) != n_train:
            parser.error(
                "--coords_paths must have the same number of entries as --features_paths."
            )
        if do_pretrain and args.pretrain_coords_path is None:
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
    train_dataset, val_dataset = build_dataset(
        args.features_paths,
        args.labels_paths,
        _none_list(args.coords_paths, n_train),
        bag_keys,
        args.dist_thr,
        val_names=val_names,
        stain_csvs=args.stain_csvs,
        stain_filter=args.stain_filter,
    )

    pretrain_dataset = None
    if do_pretrain:
        pretrain_dataset = ProcessedMILDataset(
            features_path=args.pretrain_features_path,
            labels_path=args.pretrain_labels_path,
            coords_path=args.pretrain_coords_path,
            bag_keys=bag_keys,
            dist_thr=args.dist_thr,
        )

    val_info = f" | Val bags: {len(val_dataset)}" if val_dataset is not None else ""
    print(f"Finetune train bags: {len(train_dataset)}{val_info}")
    if pretrain_dataset is not None:
        print(f"Pretrain bags: {len(pretrain_dataset)}")

    _collate = partial(collate_fn, sparse=(args.model_type not in ("deepgraphsurv", "patchgcn")))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = (
        DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
        )
        if val_dataset is not None
        else None
    )
    pretrain_loader = (
        DataLoader(
            pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        if pretrain_dataset is not None
        else None
    )

    # --- Model ---------------------------------------------------------------
    if args.model_type == "abmil":
        model = abmil_module.ABMIL(att_dim=args.att_dim)
    else:
        # Resolve feat_dim from the first available sample across datasets.
        ref_dataset = pretrain_dataset if pretrain_dataset is not None else train_dataset
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
                model, pretrain_loader, optimizer, device, args.model_type
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
            model, train_loader, optimizer, device, args.model_type
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
