"""
MIL.py — Train and validate ABMIL or DeepGraphSurv for WSI prognosis prediction.

Usage:
    python MIL.py --model_type abmil \\
        --features_path WSI/IgA/UNI2-h_feats \\
        --labels_path   WSI/IgA/labels \\
        --val_features_path WSI/IgA/UNI2-h_feats_val \\
        --val_labels_path   WSI/IgA/labels_val

    python MIL.py --model_type deepgraphsurv \\
        --features_path     WSI/IgA/UNI2-h_feats \\
        --labels_path       WSI/IgA/labels \\
        --coords_path       WSI/IgA/coords \\
        --val_features_path WSI/IgA/UNI2-h_feats_val \\
        --val_labels_path   WSI/IgA/labels_val \\
        --val_coords_path   WSI/IgA/coords_val

Labels (.npy, shape (2,)): [time_to_first_event, censoring_indicator]
"""

import os
import sys
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train ABMIL or DeepGraphSurv for WSI prognosis."
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "deepgraphsurv"],
        help="Model architecture (default: abmil).",
    )

    # --- Paths ---------------------------------------------------------------
    parser.add_argument(
        "--features_path", required=True, help="Training features folder."
    )
    parser.add_argument("--labels_path", required=True, help="Training labels folder.")
    parser.add_argument(
        "--coords_path",
        default=None,
        help="Training coords folder (deepgraphsurv only).",
    )
    parser.add_argument(
        "--val_features_path", required=True, help="Validation features folder."
    )
    parser.add_argument(
        "--val_labels_path", required=True, help="Validation labels folder."
    )
    parser.add_argument(
        "--val_coords_path",
        default=None,
        help="Validation coords folder (deepgraphsurv only).",
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", help="Folder to save checkpoints."
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        help="Final checkpoint filename (default: <model_type>_model.pth).",
    )

    # --- Hyperparameters -----------------------------------------------------
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--att_dim", type=int, default=128, help="Attention dimension.")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dim for DeepGraphSurv (default: feat_dim).",
    )
    parser.add_argument(
        "--n_layers_rep", type=int, default=1, help="DeepGraphSurv rep GCN layers."
    )
    parser.add_argument(
        "--n_layers_att", type=int, default=1, help="DeepGraphSurv att GCN layers."
    )
    parser.add_argument(
        "--K", type=int, default=5, help="Chebyshev polynomial order (deepgraphsurv)."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (deepgraphsurv)."
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

    if args.model_type == "deepgraphsurv" and args.coords_path is None:
        parser.error("--coords_path is required for deepgraphsurv.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset & DataLoader ------------------------------------------------
    bag_keys = (
        ["X", "Y", "adj", "coords"]
        if args.model_type == "deepgraphsurv"
        else ["X", "Y"]
    )

    train_dataset = ProcessedMILDataset(
        features_path=args.features_path,
        labels_path=args.labels_path,
        coords_path=args.coords_path,
        bag_keys=bag_keys,
        dist_thr=args.dist_thr,
    )
    val_dataset = ProcessedMILDataset(
        features_path=args.val_features_path,
        labels_path=args.val_labels_path,
        coords_path=args.val_coords_path,
        bag_keys=bag_keys,
        dist_thr=args.dist_thr,
    )
    print(f"Train bags: {len(train_dataset)} | Val bags: {len(val_dataset)}")

    # DeepGraphSurv needs a dense adj matrix (ChebConv uses torch.bmm).
    _collate = partial(collate_fn, sparse=(args.model_type != "deepgraphsurv"))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
    )

    # --- Model ---------------------------------------------------------------
    if args.model_type == "abmil":
        model = abmil_module.ABMIL(att_dim=args.att_dim)
    else:
        # ChebConv uses nn.Linear (not lazy), so in_shape must be given upfront.
        feat_dim = int(train_dataset[0]["X"].shape[-1])
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

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop -------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"{args.model_type}_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args.model_type
        )
        val_loss = val_epoch(model, val_loader, device, args.model_type)
        print(
            f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_ckpt = os.path.join(
                args.checkpoint_dir, f"{args.model_type}_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), epoch_ckpt)
            print(f"  → checkpoint: {epoch_ckpt}")

    final_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
