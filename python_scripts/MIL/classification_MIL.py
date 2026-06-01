"""
classification_MIL.py — Train and validate MIL models for WSI severity-score classification.

Labels: .npy files containing a single integer in [0, n_classes-1] (0-indexed, as produced by define_classification_labels.py).

Supported models:
    abmil     — Attention-Based MIL (Ilse et al., 2018)
    dsmil     — Dual-Stream MIL (Li et al., 2021)
    transmil  — Transformer MIL (Shao et al., 2021); note: does not use padding mask
    patchgcn  — Patch-based Graph CNN (Chen et al., 2021); requires --coords_paths

Two-phase training (optional):
  Phase 1 (pretrain): train on non-IgA data for --pretrain_epochs epochs (no validation).
  Phase 2 (finetune): train on IgA datasets for the remaining epochs, with validation.

Usage (ABMIL, two-phase):
    python classification_MIL.py --model_type abmil \\
        --pretrain_features_path WSI/non_IgA/UNI2-h_feats \\
        --pretrain_labels_path   WSI/non_IgA/labels \\
        --pretrain_epochs 20 \\
        --features_paths WSI/IgA/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels_classification \\
        --val_csv followup_data/classification_validation_files.csv \\
        --epochs 50

Usage (PatchGCN, requires coords):
    python classification_MIL.py --model_type patchgcn \\
        --features_paths WSI/IgA/UNI2-h_feats \\
        --labels_paths   WSI/IgA/labels_classification \\
        --coords_paths   WSI/IgA/coords \\
        --val_csv followup_data/classification_validation_files.csv

Labels (.npy): single integer in [0, n_classes-1].
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
from torchmil.models import dsmil as dsmil_module
from torchmil.models import transmil as transmil_module
from torchmil.models import patch_gcn as patch_gcn_module

_GRAPH_MODELS = {"patchgcn"}


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------
def _forward(model, batch, model_type: str) -> torch.Tensor:
    """Return logits of shape (batch_size, n_classes)."""
    if model_type == "transmil":
        # TransMIL does not accept a padding mask.
        return model(batch["X"])
    if model_type == "patchgcn":
        adj = batch["adj"]
        if adj.is_sparse:
            adj = adj.to_dense()
        return model(batch["X"], adj.float(), batch["mask"])
    # abmil, dsmil
    return model(batch["X"], batch["mask"])


def _labels(batch, device) -> torch.Tensor:
    """Return class labels (batch_size,) as long tensor. Labels are stored 0-indexed."""
    return batch["Y"].view(-1).long().to(device)


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def train_epoch(
    model, loader, optimizer, criterion, device, model_type: str
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = _forward(model, batch, model_type)
        loss = criterion(logits, _labels(batch, device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(
    model, loader, criterion, device, model_type: str
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = _forward(model, batch, model_type)
            labels = _labels(batch, device)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_filtered_bag_names(features_path, stain_csv, stain_filter):
    """Return sorted bag names matching stain_filter, or None for no filtering."""
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
    single-column file. Returns None when val_csv is None.
    """
    if val_csv is None:
        return None
    raw = pd.read_csv(val_csv, header=None, dtype=str)
    col = raw.iloc[:, 0].str.strip()
    if col.iloc[0].lower() == "file_name":
        col = col.iloc[1:]
    return set(col)


def build_dataset(
    features_paths, labels_paths, coords_paths, bag_keys, dist_thr,
    val_names=None, stain_csvs=None, stain_filter=None,
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
# Loss / accuracy logger
# ---------------------------------------------------------------------------
class LossLogger:
    """Appends metrics to a CSV every epoch and saves a two-panel plot at the end."""

    COLUMNS = ["epoch", "phase", "train_loss", "val_loss", "val_acc"]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "loss_log.csv")
        self.plot_path = os.path.join(log_dir, "loss_curves.png")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)

    def log(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, phase, f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_acc is None else f"{val_acc:.4f}",
            ])

    def save_plot(self, pretrain_epochs: int = 0):
        if not _HAS_MATPLOTLIB:
            print("matplotlib not available — skipping loss plot.")
            return
        df = pd.read_csv(self.csv_path)
        df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
        df["val_acc"] = pd.to_numeric(df["val_acc"], errors="coerce")

        has_val_loss = df["val_loss"].notna().any()
        has_val_acc = df["val_acc"].notna().any()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss, ax_acc = axes

        for ax in axes:
            if pretrain_epochs > 0:
                ax.axvspan(0.5, pretrain_epochs + 0.5, alpha=0.08, color="gray",
                           label="pretrain")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)

        ax_loss.plot(df["epoch"], df["train_loss"], label="train loss")
        if has_val_loss:
            val_rows = df["val_loss"].notna()
            ax_loss.plot(df.loc[val_rows, "epoch"], df.loc[val_rows, "val_loss"],
                         linestyle="--", label="val loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        if has_val_acc:
            val_rows = df["val_acc"].notna()
            ax_acc.plot(df.loc[val_rows, "epoch"], df.loc[val_rows, "val_acc"],
                        color="tab:green", label="val accuracy")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_type: str, feat_dim: int, n_classes: int, args) -> nn.Module:
    """Instantiate model and replace its 1-output classifier with n_classes output.

    All torchmil models use self.classifier = LazyLinear(..., 1) as the final layer
    and squeeze(-1) in forward, which is a no-op when n_classes > 1.
    """
    if model_type == "abmil":
        model = abmil_module.ABMIL(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            gated=args.gated,
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

    # Replace the single-output head with a n_classes head.
    # squeeze(-1) in torchmil forwards is a no-op for tensors with last dim > 1,
    # so outputs will be (batch_size, n_classes) as expected by CrossEntropyLoss.
    model.classifier = nn.LazyLinear(n_classes)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train MIL models for WSI severity-score classification."
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "dsmil", "transmil", "patchgcn"],
        help="Model architecture (default: abmil).",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=5,
        help="Number of severity classes (default: 5).",
    )

    # --- Phase 1: pretrain on non-IgA (no val files) -------------------------
    parser.add_argument(
        "--pretrain_features_path", default=None,
        help="Non-IgA features folder for pretraining.",
    )
    parser.add_argument(
        "--pretrain_labels_path", default=None,
        help="Non-IgA labels folder for pretraining.",
    )
    parser.add_argument(
        "--pretrain_coords_path", default=None,
        help="Non-IgA coords folder for pretraining (patchgcn only).",
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=0,
        help="Epochs on non-IgA before switching to IgA datasets (default: 0).",
    )

    # --- Phase 2: finetune ---------------------------------------------------
    parser.add_argument(
        "--features_paths", nargs="+", required=True,
        help="Training features folder(s).",
    )
    parser.add_argument(
        "--labels_paths", nargs="+", required=True,
        help="Training labels folder(s).",
    )
    parser.add_argument(
        "--coords_paths", nargs="+", default=None,
        help="Training coords folder(s) (patchgcn only).",
    )
    parser.add_argument(
        "--val_csv", default=None,
        help=(
            "CSV listing validation slide basenames ('file_name' column or headerless). "
            "The same feature/label directories are used; bags are split at load time."
        ),
    )

    # --- Stain filtering -----------------------------------------------------
    parser.add_argument(
        "--stain_filter", default=None,
        help="Keep only bags with this value in the 'Stain' column (e.g. PAS).",
    )
    parser.add_argument(
        "--stain_csvs", nargs="+", default=None,
        help="Path to labels_combined.csv per --features_paths entry; use 'none' to skip.",
    )

    # --- Output --------------------------------------------------------------
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Folder to save checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name", default=None,
        help="Final checkpoint filename (default: <model_type>_model.pth).",
    )
    parser.add_argument(
        "--log_dir", default=None,
        help="Folder for loss_log.csv and loss_curves.png (default: checkpoint_dir).",
    )

    # --- Hyperparameters -----------------------------------------------------
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Total training epochs including pretrain (default: 50).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--att_dim", type=int, default=128,
        help="Attention dim (TransMIL paper uses 512; default: 128).",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Hidden dim for PatchGCN (default: feat_dim).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate (dsmil / transmil / patchgcn).",
    )
    parser.add_argument(
        "--dist_thr", type=float, default=1.5,
        help="Adjacency distance threshold (patchgcn).",
    )

    # ABMIL-specific
    parser.add_argument(
        "--gated", action="store_true",
        help="Use gated attention in ABMIL (default: False).",
    )

    # DSMIL-specific
    parser.add_argument(
        "--nonlinear_q", action="store_true",
        help="Use nonlinear query projection in DSMIL.",
    )
    parser.add_argument(
        "--nonlinear_v", action="store_true",
        help="Use nonlinear value projection in DSMIL.",
    )

    # TransMIL-specific
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of transformer layers (transmil).",
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Number of attention heads (transmil).",
    )

    # PatchGCN-specific
    parser.add_argument(
        "--n_gcn_layers", type=int, default=4,
        help="Number of GCN layers (patchgcn).",
    )
    parser.add_argument(
        "--mlp_depth", type=int, default=1,
        help="MLP depth after GCN (patchgcn).",
    )

    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save a checkpoint every N epochs (0 = only final).",
    )

    args = parser.parse_args()

    # --- Validate argument combinations --------------------------------------
    n_train = len(args.features_paths)
    if len(args.labels_paths) != n_train:
        parser.error("--features_paths and --labels_paths must have the same number of entries.")

    do_pretrain = args.pretrain_features_path is not None
    if do_pretrain:
        if args.pretrain_labels_path is None:
            parser.error("--pretrain_labels_path is required when --pretrain_features_path is set.")
        if args.pretrain_epochs <= 0:
            parser.error("--pretrain_epochs must be > 0 when --pretrain_features_path is set.")
    if args.pretrain_epochs > 0 and not do_pretrain:
        parser.error("--pretrain_features_path is required when --pretrain_epochs > 0.")

    if args.model_type == "patchgcn":
        if args.coords_paths is None:
            parser.error("--coords_paths is required for patchgcn.")
        if len(args.coords_paths) != n_train:
            parser.error("--coords_paths must have the same number of entries as --features_paths.")
        if do_pretrain and args.pretrain_coords_path is None:
            parser.error("--pretrain_coords_path is required for patchgcn pretraining.")

    if args.stain_csvs is not None:
        if args.stain_filter is None:
            parser.error("--stain_filter is required when --stain_csvs is set.")
        if len(args.stain_csvs) != n_train:
            parser.error("--stain_csvs must have the same number of entries as --features_paths.")

    finetune_epochs = args.epochs - args.pretrain_epochs
    if finetune_epochs <= 0:
        parser.error("--pretrain_epochs must be less than --epochs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_graph = args.model_type in _GRAPH_MODELS
    bag_keys = ["X", "Y", "adj", "coords"] if is_graph else ["X", "Y"]

    def _none_list(paths, n):
        return paths if paths is not None else [None] * n

    # --- Build datasets -------------------------------------------------------
    val_names = load_val_names(args.val_csv)
    train_dataset, val_dataset = build_dataset(
        args.features_paths, args.labels_paths,
        _none_list(args.coords_paths, n_train),
        bag_keys, args.dist_thr,
        val_names=val_names,
        stain_csvs=args.stain_csvs, stain_filter=args.stain_filter,
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

    # Dense adj for patchgcn; sparse (COO) for attention-based models.
    _collate = partial(collate_fn, sparse=(not is_graph))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
        if val_dataset is not None else None
    )
    pretrain_loader = (
        DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
        if pretrain_dataset is not None else None
    )

    # --- Model ---------------------------------------------------------------
    ref_dataset = pretrain_dataset if pretrain_dataset is not None else train_dataset
    feat_dim = int(ref_dataset[0]["X"].shape[-1])
    model = build_model(args.model_type, feat_dim, args.n_classes, args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
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
                model, pretrain_loader, optimizer, criterion, device, args.model_type,
            )
            logger.log(epoch, "pretrain", train_loss)
            print(f"[Pretrain {epoch:>3d}/{args.pretrain_epochs}] train_loss={train_loss:.4f}")

            if args.save_every > 0 and epoch % args.save_every == 0:
                ckpt = os.path.join(
                    args.checkpoint_dir, f"{args.model_type}_pretrain_epoch{epoch}.pth"
                )
                torch.save(model.state_dict(), ckpt)
                print(f"  → checkpoint: {ckpt}")

    # --- Phase 2: finetune ---------------------------------------------------
    print(f"\n--- Finetune phase: {finetune_epochs} epochs on IgA datasets ---")
    for epoch in range(1, finetune_epochs + 1):
        global_epoch = epoch + args.pretrain_epochs
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.model_type,
        )

        if val_loader is not None:
            val_loss, val_acc = val_epoch(
                model, val_loader, criterion, device, args.model_type,
            )
            logger.log(global_epoch, "finetune", train_loss, val_loss, val_acc)
            print(
                f"[Epoch {global_epoch:>3d}/{args.epochs}]"
                f" train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )
        else:
            logger.log(global_epoch, "finetune", train_loss)
            print(f"[Epoch {global_epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}")

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
