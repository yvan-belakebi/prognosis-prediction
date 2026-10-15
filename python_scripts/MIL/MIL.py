"""
MIL.py — Train and validate ABMIL, TransMIL, DeepGraphSurv or PatchGCN for WSI
prognosis prediction.

Data: one flat features folder (--features_path, <slide>.h5), the flat .npy labels
from define_labels.py (--labels_path), and its labels CSV (--labels_csv), which
supplies each slide's biopsy (for --biopsy_sampling / --max_biopsies), its stain
(for --stains) and its train/val split.

Usage (ABMIL on PAS slides):
    python python_scripts/MIL/MIL.py --model_type abmil \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels \\
        --labels_csv    label_csvs/labels.csv \\
        --stains PAS --epochs 50

Usage (DeepGraphSurv — coords auto-read from features .h5, no --coords_path needed):
    python python_scripts/MIL/MIL.py --model_type deepgraphsurv \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels \\
        --labels_csv    label_csvs/labels.csv \\
        --stains PAS "HES|HE" \\
        --dropout 0.1 --save_every 5 --batch_size 4

Usage (TransMIL — self-attention pooling, no coords/adjacency needed):
    python python_scripts/MIL/MIL.py --model_type transmil \\
        --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \\
        --labels_path   WSI/IgA/trident/labels \\
        --labels_csv    label_csvs/labels.csv \\
        --n_layers 2 --n_heads 4 --dropout 0.1 --batch_size 4
    # TransMIL attends over every instance and takes no padding mask, so keep
    # --batch_size small (or use --max_patches) on large bags.

Resuming: every epoch writes <checkpoint_name>_last.pth (weights + optimizer +
epoch + best val C-index). To continue an interrupted or finished run, repeat the
same command with --resume and the total number of epochs wanted:
    python python_scripts/MIL/MIL.py <same arguments> --epochs 80 \\
        --resume checkpoints/deepgraphsurv_model_last.pth

Usage (diagnosis-code late fusion): first regenerate labels with the code +
vocabulary, then pass --use_diagnosis (works with any --model_type):
    python python_scripts/prepare_for_MIL/define_labels.py --with_diagnosis ...
    python python_scripts/MIL/MIL.py --model_type abmil --use_diagnosis \\
        --features_path ... --labels_path ... --labels_csv label_csvs/labels.csv \\
        --diagnosis_codes_json label_csvs/diagnosis_codes.json --diag_dim 16
    # A learned code embedding is concatenated to the pooled bag vector before
    # the risk head. --diag_dim must match at eval/inference time.

Labels (.npy): [time_to_first_event, censoring_indicator], with an optional
third element [.., .., diagnosis_code] when trained with --use_diagnosis
(written by define_labels.py --with_diagnosis).
"""

import os
import sys
import json
import argparse
from functools import partial

import pandas as pd
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

from torchmil.data import collate_fn
from torchmil.models import abmil as abmil_module
from torchmil.models import transmil as transmil_module
from torchmil.models import deepgraphsurv as dgs_module
from torchmil.models import patch_gcn as patch_gcn_module

from mil_utils import (
    load_authorized_slides,
    load_labels_table,
    make_collate_fn,
    GpuMemLogger,
    build_dataset,
    BiopsySampler,
    LossLogger,
)
from late_fusion import LateFusionSurv

_evaluate_dir = os.path.abspath(os.path.join(_script_dir, "..", "evaluate_MIL"))
if _evaluate_dir not in sys.path:
    sys.path.insert(0, _evaluate_dir)
from evaluate_survival import aggregate_by_biopsy, concordance_index  # noqa: E402


class SurvivalLogger(LossLogger):
    """Loss log with the biopsy-level validation C-index, plotted in a second panel."""

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_cindex"]

    def log(self, epoch, train_loss, val_loss=None, val_cindex=None):
        self._append(
            [
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_cindex is None else f"{val_cindex:.4f}",
            ]
        )

    def save_plot(self):
        import matplotlib.pyplot as plt

        df = pd.read_csv(self.csv_path)
        for col in ("val_loss", "val_cindex"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        val = df[df["val_loss"].notna()]

        fig, (ax_loss, ax_c) = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss.plot(df["epoch"], df["train_loss"], label="train loss (per batch)")
        if not val.empty:
            ax_loss.plot(val["epoch"], val["val_loss"], linestyle="--",
                         label="val loss (whole set)")
            ax_c.plot(val["epoch"], val["val_cindex"], color="tab:green",
                      label="val C-index (per biopsy)")
            ax_c.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            ax_c.legend()
        ax_loss.set_ylabel("Cox loss")
        ax_loss.legend()
        ax_c.set_ylabel("C-index")
        for ax in (ax_loss, ax_c):
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")


# ---------------------------------------------------------------------------
# Resumable checkpoints
# ---------------------------------------------------------------------------
# Arguments that define the weight shapes: resuming with any of them changed
# cannot load the weights, so it is refused with a clear message.
_ARCH_ARGS = (
    "model_type", "att_dim", "hidden_dim", "n_layers", "n_heads", "n_layers_rep",
    "n_layers_att", "K", "n_gcn_layers", "mlp_depth", "use_diagnosis", "diag_dim",
)


def save_last_checkpoint(path, model, optimizer, epoch, best_cindex, best_epoch, args):
    """Write everything needed to resume after ``epoch``, atomically.

    The weights-only checkpoints (_epochN, _best, final) stay as they are for the
    evaluation scripts; this one adds the optimizer state (Adam's moment
    estimates), the epoch, the best val C-index so far, the RNG states and the
    run's arguments. It is written to a temporary file and renamed, so a crash
    during the save leaves the previous checkpoint intact.
    """
    import random

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_cindex": best_cindex,
        "best_epoch": best_epoch,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_last_checkpoint(path, model, optimizer, args, device, parser):
    """Restore a save_last_checkpoint file; return (epoch, best_cindex, best_epoch)."""
    import random

    state = torch.load(path, map_location=device, weights_only=False)
    saved = state["args"]
    changed = [k for k in _ARCH_ARGS if saved.get(k) != getattr(args, k)]
    if changed:
        parser.error(
            "--resume: the checkpoint was trained with a different architecture ("
            + ", ".join(f"{k}={saved.get(k)!r}, now {getattr(args, k)!r}" for k in changed)
            + "). Pass the same values to resume."
        )
    ignore = {"resume", "epochs", "num_workers", "mem_debug"}
    for k, v in sorted(vars(args).items()):
        if k not in ignore and k not in _ARCH_ARGS and k in saved and saved[k] != v:
            print(f"Warning: --{k} was {saved[k]!r} in the checkpoint, now {v!r}.")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    # The optimizer state carries the old learning rate and weight decay; the
    # resume command line wins, so either can be changed mid-run. Checkpoints
    # from before --weight_decay were saved by Adam, whose state carries
    # decoupled_weight_decay=False; it is forced back on so the decay stays
    # AdamW's (PyTorch versions without the key always decouple in AdamW).
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["weight_decay"] = args.weight_decay
        if "decoupled_weight_decay" in group:
            group["decoupled_weight_decay"] = True
    # map_location moved every tensor to `device`, but the CPU generator only
    # accepts a CPU ByteTensor.
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    return state["epoch"], state["best_cindex"], state["best_epoch"]


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
    # transmil pools with self-attention over all instances and takes no mask.
    if model_type == "transmil":
        return model(batch["X"])
    # Graph models: the adjacency stays sparse COO (their layers use torch.bmm,
    # which accepts it). A dense one is B x N x N floats — 1.5 GB for a single
    # 20k-patch slide — which is what ran out of memory. Only the float64 values
    # from numpy are cast.
    return model(batch["X"], batch["adj"].float(), batch["mask"])


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------
def train_epoch(
    model, loader, optimizer, device, risk_fn, accumulation_steps: int = 1,
    mem_logger=None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        if mem_logger is not None:
            mem_logger.step_start()
        try:
            risk = risk_fn(model, batch)
            loss = cox_ph_loss(risk, batch["Y"][:, 0].float(), batch["Y"][:, 1].float())
            # Scale so that accumulated gradients match a single full-batch step.
            # Note: for Cox PH the risk set is per micro-batch, which is approximate
            # but acceptable in practice.
            (loss / accumulation_steps).backward()
        except torch.OutOfMemoryError:
            if mem_logger is not None:
                mem_logger.report_oom(i, batch)
            raise
        total_loss += loss.item()
        if mem_logger is not None:
            mem_logger.log(i, batch)
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / len(loader)


def val_epoch(model, loader, device, risk_fn, bag_names, table):
    """Return (val_loss, val_cindex) over the whole validation set.

    The Cox loss uses the whole val set as one risk set instead of averaging
    per-batch losses: with the small batches graph models need, most batches
    hold zero or one event, so a per-batch average is mostly noise. The C-index
    is computed per biopsy (slide risks averaged, as in evaluate_survival.py).

    ``bag_names`` must be in loader order (shuffle=False). Patch subsampling
    (--max_patches) runs under a fixed seed, so every epoch scores the same
    patches; the training RNG stream is left untouched.
    """
    model.eval()
    risks, labels = [], []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for batch in loader:
            batch = batch.to(device)
            risks.append(risk_fn(model, batch).reshape(-1).float().cpu())
            labels.append(batch["Y"][:, :2].float().cpu())
    risk, y = torch.cat(risks), torch.cat(labels)
    loss = cox_ph_loss(risk, y[:, 0], y[:, 1]).item()
    b_risk, b_time, b_event, _ = aggregate_by_biopsy(
        risk.numpy(), y[:, 0].numpy(), y[:, 1].numpy(), bag_names, table
    )
    return loss, concordance_index(b_risk, b_time, b_event)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train ABMIL / TransMIL / DeepGraphSurv / PatchGCN for WSI prognosis."
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "transmil", "deepgraphsurv", "patchgcn"],
        help="Model architecture (default: abmil).",
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

    # --- Diagnosis-code late fusion -----------------------------------------
    parser.add_argument(
        "--use_diagnosis",
        action="store_true",
        help=(
            "Condition the model on a per-bag diagnosis code via late fusion "
            "(concat a learned code embedding to the pooled bag vector before "
            "the risk head). Requires 3-element labels [time, event, code_idx] "
            "and --diagnosis_codes_json (see define_labels.py --with_diagnosis)."
        ),
    )
    parser.add_argument(
        "--diagnosis_codes_json",
        default="label_csvs/diagnosis_codes.json",
        help="JSON mapping produced by define_labels.py; provides the code "
             "vocabulary size (n_codes) for the diagnosis embedding.",
    )
    parser.add_argument(
        "--diag_dim",
        type=int,
        default=16,
        help="Diagnosis code embedding dimension (default: 16).",
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
        help="Training epochs (default: 50).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes that load bags and build graphs while the "
             "GPU trains (0 = load in the main process).",
    )
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
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay, a regulariser against overfitting (0 = plain Adam). "
             "Typical values 1e-4 to 1e-2.",
    )
    parser.add_argument("--att_dim", type=int, default=128, help="Attention dimension.")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dim for DeepGraphSurv / PatchGCN (default: feat_dim).",
    )
    # TransMIL-specific
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Transformer layers (transmil)."
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Attention heads (transmil)."
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
        help="Dropout rate (transmil / deepgraphsurv / patchgcn).",
    )
    parser.add_argument(
        "--dist_thr",
        type=float,
        default=1.5,
        help="Adjacency distance threshold (deepgraphsurv / patchgcn), in patches: "
             "TRIDENT coords are divided by their patch spacing, so 1.5 links each "
             "patch to its 8 neighbours.",
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
        "--mem_debug",
        action="store_true",
        help=(
            "Write a per-step CUDA memory trace to <log_dir>/gpu_mem.csv and, on "
            "an OOM, dump the failing batch shape plus the other processes "
            "holding GPU memory. Use with monitor_gpu.py."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 = only final).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a <checkpoint_name>_last.pth (written every epoch): restores "
             "the weights, optimizer, epoch and best val C-index, continues up to "
             "--epochs, and appends to the existing loss log. Pass the same "
             "architecture arguments as the original run.",
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
            "Maximum number of training biopsies (all slides of a selected biopsy "
            "are kept). Has no effect on the validation set."
        ),
    )

    args = parser.parse_args()
    torch.cuda.empty_cache()

    # --- Validate argument combinations --------------------------------------
    if args.model_type in ("deepgraphsurv", "patchgcn") and args.coords_path is None:
        if args.file_ext == ".h5":
            # Coords are embedded in the features .h5; reuse the same directory.
            args.coords_path = args.features_path
        else:
            parser.error(f"--coords_path is required for {args.model_type}.")

    if args.model_type == "transmil" and args.att_dim % args.n_heads != 0:
        parser.error(
            f"--att_dim ({args.att_dim}) must be divisible by --n_heads "
            f"({args.n_heads}) for transmil."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bag_keys = (
        ["X", "Y", "adj", "coords"]
        if args.model_type in ("deepgraphsurv", "patchgcn")
        else ["X", "Y"]
    )

    # --- Build datasets -------------------------------------------------------
    table = load_labels_table(args.labels_csv)
    train_dataset, val_dataset, _, _ = build_dataset(
        args.features_path,
        args.labels_path,
        args.coords_path,
        bag_keys,
        args.dist_thr,
        table,
        stains=args.stains,
        max_biopsies=args.max_biopsies,
        file_ext=args.file_ext,
        authorized_slides=load_authorized_slides(args.authorized_slides_csv),
    )
    val_info = f" | Val bags: {len(val_dataset)}" if val_dataset is not None else ""
    print(f"Train bags: {len(train_dataset)}{val_info}")

    # Subsample patches before adj is built to avoid OOM on large slides.
    # sparse=True keeps the graph models' adjacency sparse (see _forward).
    _collate = make_collate_fn(
        partial(collate_fn, sparse=True), max_patches=args.max_patches
    )

    # Workers read the .h5 bags and build the graphs in parallel with the GPU
    # step; with num_workers=0 that all happened between steps and the GPU sat
    # idle. The val loader's workers are not persistent: they are re-created
    # each epoch from the fixed seed in val_epoch, so validation stays
    # deterministic.
    _train_sampler = BiopsySampler(train_dataset, table) if args.biopsy_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(_train_sampler is None),
        sampler=_train_sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
            num_workers=args.num_workers,
        )
        if val_dataset is not None
        else None
    )

    # --- Model ---------------------------------------------------------------
    if args.model_type == "abmil":
        model = abmil_module.ABMIL(att_dim=args.att_dim)
    else:
        feat_dim = int(train_dataset[0]["X"].shape[-1])
        if args.model_type == "transmil":
            model = transmil_module.TransMIL(
                in_shape=(feat_dim,),
                att_dim=args.att_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dropout=args.dropout,
            )
        elif args.model_type == "deepgraphsurv":
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

    # --- Optional diagnosis-code late fusion ---------------------------------
    if args.use_diagnosis:
        n_label = int(train_dataset[0]["Y"].numel())
        if n_label < 3:
            parser.error(
                "--use_diagnosis requires 3-element labels [time, event, code_idx], "
                f"but the label files hold {n_label} values. Regenerate labels with "
                "define_labels.py --with_diagnosis."
            )
        with open(args.diagnosis_codes_json) as f:
            codes_meta = json.load(f)
        n_codes = codes_meta.get("n_codes") or (max(codes_meta["codes"].values()) + 1)
        model = LateFusionSurv(
            model, args.model_type, n_codes=n_codes, diag_dim=args.diag_dim
        )
        print(f"Late fusion enabled: {n_codes} diagnosis codes, diag_dim={args.diag_dim}")
        risk_fn = lambda m, batch: m(batch)  # noqa: E731
    else:
        risk_fn = lambda m, batch: _forward(m, batch, args.model_type)  # noqa: E731

    model = model.to(device)
    # AdamW: weight decay applied directly to the weights, not added to the
    # gradient as Adam's L2 term is (which Adam's per-weight scaling then
    # mostly cancels). With --weight_decay 0 it is identical to Adam.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = args.checkpoint_name or f"{args.model_type}_model.pth"
    best_path = os.path.join(
        args.checkpoint_dir, f"{os.path.splitext(checkpoint_name)[0]}_best.pth"
    )
    last_path = os.path.join(
        args.checkpoint_dir, f"{os.path.splitext(checkpoint_name)[0]}_last.pth"
    )
    best_cindex, best_epoch, start_epoch = -1.0, None, 1
    if args.resume:
        done, best_cindex, best_epoch = load_last_checkpoint(
            args.resume, model, optimizer, args, device, parser
        )
        start_epoch = done + 1
        best_info = (
            f", best val C-index {best_cindex:.4f} at epoch {best_epoch}"
            if best_epoch is not None else ""
        )
        print(f"Resumed from {args.resume}: {done} epochs done{best_info}.")
        if start_epoch > args.epochs:
            print(f"Nothing to do: --epochs {args.epochs} already reached.")

    log_dir = args.log_dir or args.checkpoint_dir
    logger = SurvivalLogger(log_dir, resume_epoch=start_epoch - 1 if args.resume else None)
    print(f"Loss log: {logger.csv_path}")

    mem_logger = None
    if args.mem_debug:
        if device.type != "cuda":
            parser.error("--mem_debug requires a CUDA device.")
        mem_logger = GpuMemLogger(log_dir)
        print(f"GPU memory trace: {mem_logger.path}")

    # --- Training ------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        if mem_logger is not None:
            mem_logger.set_phase(f"epoch{epoch}")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            risk_fn,
            accumulation_steps=args.accumulation_steps,
            mem_logger=mem_logger,
        )

        if val_loader is not None:
            val_loss, val_cindex = val_epoch(
                model, val_loader, device, risk_fn, val_dataset.bag_names, table
            )
            logger.log(epoch, train_loss, val_loss, val_cindex)
            improved = val_cindex > best_cindex
            if improved:
                best_cindex, best_epoch = val_cindex, epoch
                torch.save(model.state_dict(), best_path)
            print(
                f"[Epoch {epoch:>3d}/{args.epochs}]"
                f" train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                f"  val_C={val_cindex:.4f}" + ("  *best" if improved else "")
            )
        else:
            logger.log(epoch, train_loss)
            print(f"[Epoch {epoch:>3d}/{args.epochs}] train_loss={train_loss:.4f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"{args.model_type}_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt}")

        save_last_checkpoint(
            last_path, model, optimizer, epoch, best_cindex, best_epoch, args
        )

    final_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")
    if best_epoch is not None:
        print(
            f"Best val C-index {best_cindex:.4f} at epoch {best_epoch} → {best_path}"
        )

    logger.save_plot()


if __name__ == "__main__":
    main()
