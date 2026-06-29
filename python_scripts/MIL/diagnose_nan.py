"""
diagnose_nan.py — Locate the source of NaN losses in regression_MIL.py.

Walks the exact train/val datasets built by build_dataset() and, for every bag,
checks the feature tensor and a model forward pass for NaN/Inf. Prints the
offending bag name(s) so you can quarantine or re-extract them.

Usage (same data args as regression_MIL.py):
    python python_scripts/MIL/diagnose_nan.py \
        --model_type transmil \
        --features_paths WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 \
        --labels_paths   WSI/IgA/trident/labels_regression \
        --val_csv validation_files_csvs/regression_validation_files.csv \
        --att_dim 128 --dropout 0.1
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

from torchmil.data import collate_fn  # noqa: E402
from torchmil.models import transmil as transmil_module  # noqa: E402

from mil_utils import (  # noqa: E402
    discover_bags,
    load_val_names,
    build_dataset,
    get_bag_names,
)
from regression_MIL import scan_labels, build_model, _forward  # noqa: E402


def _flag(t):
    """Return (has_nan, has_inf) for a tensor, densifying sparse tensors."""
    if t.is_sparse:
        t = t.to_dense()
    return bool(torch.isnan(t).any()), bool(torch.isinf(t).any())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="transmil")
    p.add_argument("--features_paths", nargs="+", required=True)
    p.add_argument("--labels_paths", nargs="+", required=True)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--file_ext", default=".h5")
    p.add_argument("--att_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--gated", action="store_true")
    p.add_argument("--nonlinear_q", action="store_true")
    p.add_argument("--nonlinear_v", action="store_true")
    p.add_argument("--n_gcn_layers", type=int, default=4)
    p.add_argument("--mlp_depth", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--dist_thr", type=float, default=1.5)
    args = p.parse_args()

    n = len(args.features_paths)
    val_names = load_val_names(args.val_csv)
    train_ds, val_ds, train_labels, val_labels = build_dataset(
        args.features_paths,
        args.labels_paths,
        [None] * n,
        ["X", "Y"],
        args.dist_thr,
        val_names=val_names,
        scan_labels_fn=scan_labels,
        file_ext=args.file_ext,
    )

    # --- Label sanity (val side is not printed by the training script) ---
    for tag, lbl in (("train", train_labels), ("val", val_labels)):
        if lbl is None:
            continue
        bad = np.where(~np.isfinite(lbl))[0]
        print(f"{tag} labels: {len(lbl)} | non-finite: {len(bad)}")
        if len(bad):
            print(f"  -> first bad indices: {bad[:10].tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = int((val_ds or train_ds)[0]["X"].shape[-1])
    model = build_model(args.model_type, feat_dim, args).to(device)
    model.eval()

    for tag, ds in (("val", val_ds), ("train", train_ds)):
        if ds is None:
            continue
        names = get_bag_names(ds)
        print(f"\n=== Scanning {tag} set ({len(ds)} bags) ===")
        n_feat_bad = n_fwd_bad = 0
        for i in range(len(ds)):
            bag = ds[i]
            name = names[i] if i < len(names) else f"idx{i}"
            x = bag["X"]
            nan_x, inf_x = _flag(x)
            n_patches = x.shape[0]
            if nan_x or inf_x or n_patches == 0:
                n_feat_bad += 1
                print(
                    f"  [FEAT] {name}: patches={n_patches} nan={nan_x} inf={inf_x}"
                )
                continue
            batch = collate_fn([bag], sparse=True).to(device)
            with torch.no_grad():
                pred = _forward(model, batch, args.model_type)
            nan_p, inf_p = _flag(pred.cpu())
            if nan_p or inf_p:
                n_fwd_bad += 1
                print(
                    f"  [FWD ] {name}: patches={n_patches} "
                    f"pred_nan={nan_p} pred_inf={inf_p}"
                )
        print(
            f"{tag}: {n_feat_bad} bags with bad features, "
            f"{n_fwd_bad} bags with NaN/Inf forward output"
        )


if __name__ == "__main__":
    main()
