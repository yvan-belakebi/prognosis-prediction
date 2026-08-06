"""
inference_MIL.py  —  Run ABMIL or DeepGraphSurv prognosis inference on a single WSI.

Usage:
    # ABMIL
    python inference_MIL.py \\
        --features_folder PATH_TO_NPY_FOLDER \\
        --checkpoint      PATH_TO_CHECKPOINT.pth \\
        [--model_type abmil] [--att_dim 128]

    # DeepGraphSurv
    python inference_MIL.py \\
        --features_folder PATH_TO_NPY_FOLDER \\
        --coords_folder   PATH_TO_COORDS_FOLDER \\
        --checkpoint      PATH_TO_CHECKPOINT.pth \\
        --model_type      deepgraphsurv \\
        [--dist_thr 1.5] [--att_dim 128] [--hidden_dim 128]

Features folder: one .npy file of shape (N_patches, feat_dim), OR multiple files of
shape (feat_dim,) / (1, feat_dim) that are stacked automatically.
Coords  folder:  same convention — one .npy file of shape (N_patches, 2) expected.

Output: risk score printed to stdout (higher = worse prognosis).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Locate the local torchmil package
# ---------------------------------------------------------------------------
def _find_torchmil_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    candidates = [
        os.path.join(project_root, "torchmil"),
        os.path.join(os.path.dirname(project_root), "torchmil"),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "torchmil")):
            return path
    return None


_torchmil_root = _find_torchmil_root()
if _torchmil_root and _torchmil_root not in sys.path:
    sys.path.insert(0, _torchmil_root)

try:
    from torchmil.models import abmil as _abmil_module
    from torchmil.models import deepgraphsurv as _dgs_module
    from torchmil.utils import build_adj, normalize_adj, add_self_loops

    _TORCHMIL_AVAILABLE = True
except ImportError:
    _TORCHMIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal ABMIL fallback (used only when torchmil is unavailable)
# ---------------------------------------------------------------------------
class _FallbackABMIL(nn.Module):
    def __init__(self, att_dim: int = 128):
        super().__init__()
        self.attention_V = nn.LazyLinear(att_dim)
        self.attention_w = nn.Linear(att_dim, 1, bias=False)
        self.classifier = nn.LazyLinear(1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        A = self.attention_w(torch.tanh(self.attention_V(X)))  # (B, N, 1)
        if mask is not None:
            A = A.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
        A = torch.softmax(A, dim=1)
        Z = (A * X).sum(dim=1)
        return self.classifier(Z).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature / coord loading
# ---------------------------------------------------------------------------
def _load_npy_folder(folder: str, name: str) -> np.ndarray:
    """Stack all .npy files in *folder* into a 2-D array (N, D)."""
    npy_files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {name} folder: {folder}")
    arrays = []
    for fname in npy_files:
        arr = np.load(os.path.join(folder, fname))
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def load_features(folder: str) -> np.ndarray:
    return _load_npy_folder(folder, "features")


def load_coords(folder: str) -> np.ndarray:
    return _load_npy_folder(folder, "coords")


# ---------------------------------------------------------------------------
# Adjacency construction from coordinates
# ---------------------------------------------------------------------------
def build_dense_adj(coords: np.ndarray, dist_thr: float) -> torch.Tensor:
    """
    Build a normalized, dense adjacency matrix from patch coordinates.

    Returns tensor of shape (1, N, N) ready for DeepGraphSurv forward().
    """
    n = len(coords)
    edge_index, edge_weight = build_adj(coords, None, dist_thr=dist_thr)
    if n == 1 or edge_index.size == 0:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, n)
    norm_weight = normalize_adj(edge_index, edge_weight, n_nodes=n)

    adj = torch.zeros(n, n, dtype=torch.float32)
    if edge_index.size > 0:
        rows = torch.from_numpy(edge_index[0]).long()
        cols = torch.from_numpy(edge_index[1]).long()
        vals = torch.from_numpy(norm_weight).float()
        adj[rows, cols] = vals

    return adj.unsqueeze(0)  # (1, N, N)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ABMIL / DeepGraphSurv prognosis inference for a single WSI."
    )
    parser.add_argument(
        "--features_folder",
        required=True,
        help="Folder containing .npy feature file(s) for one WSI.",
    )
    parser.add_argument(
        "--coords_folder",
        default=None,
        help="Folder containing .npy coords file(s) for one WSI (required for deepgraphsurv).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "deepgraphsurv"],
        help="Model architecture: 'abmil' (default) or 'deepgraphsurv'.",
    )
    parser.add_argument("--att_dim", type=int, default=128, help="Attention dimension.")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dim (deepgraphsurv, default: feat_dim).")
    parser.add_argument("--n_layers_rep", type=int, default=1, help="Rep GCN layers (deepgraphsurv).")
    parser.add_argument("--n_layers_att", type=int, default=1, help="Att GCN layers (deepgraphsurv).")
    parser.add_argument("--K", type=int, default=5, help="Chebyshev order (deepgraphsurv).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (deepgraphsurv, ignored at inference).")
    parser.add_argument("--dist_thr", type=float, default=1.5, help="Adjacency distance threshold (deepgraphsurv).")
    parser.add_argument(
        "--return_att",
        action="store_true",
        help="Print per-patch attention weights (top 10).",
    )
    # --- Diagnosis-code late fusion (must match the trained checkpoint) -------
    parser.add_argument(
        "--use_diagnosis",
        action="store_true",
        help="Load a late-fusion checkpoint that was conditioned on a diagnosis "
        "code. Requires --diagnosis or --diagnosis_code and --diagnosis_codes_json.",
    )
    parser.add_argument(
        "--diagnosis",
        default=None,
        help="Diagnosis string for this WSI (mapped to its code via "
        "--diagnosis_codes_json; unknown/unlisted maps to 0).",
    )
    parser.add_argument(
        "--diagnosis_code",
        type=int,
        default=None,
        help="Diagnosis code index for this WSI (overrides --diagnosis).",
    )
    parser.add_argument(
        "--diagnosis_codes_json",
        default="label_csvs/diagnosis_codes.json",
        help="JSON vocabulary written by define_labels.py --with_diagnosis.",
    )
    parser.add_argument(
        "--diag_dim", type=int, default=16, help="Diagnosis embedding dim (must match training)."
    )
    args = parser.parse_args()

    if args.model_type == "deepgraphsurv" and args.coords_folder is None:
        parser.error("--coords_folder is required for deepgraphsurv.")

    if args.use_diagnosis and not _TORCHMIL_AVAILABLE:
        parser.error("torchmil is required for --use_diagnosis late fusion.")
    if args.use_diagnosis and args.diagnosis is None and args.diagnosis_code is None:
        parser.error("--use_diagnosis requires --diagnosis or --diagnosis_code.")

    if not _TORCHMIL_AVAILABLE and args.model_type == "deepgraphsurv":
        parser.error("torchmil is required for deepgraphsurv but could not be imported.")

    if not _TORCHMIL_AVAILABLE:
        print("Warning: torchmil not found — using built-in ABMIL fallback.", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load features first (feat_dim needed by DeepGraphSurv constructor) --
    features = load_features(args.features_folder)
    n_patches, feat_dim = features.shape
    print(f"Loaded {n_patches} patches × {feat_dim}-dim features.")

    # --- Build model ---------------------------------------------------------
    if args.model_type == "abmil":
        model = (
            _abmil_module.ABMIL(att_dim=args.att_dim)
            if _TORCHMIL_AVAILABLE
            else _FallbackABMIL(args.att_dim)
        )
    else:
        # ChebConv uses nn.Linear (not lazy) — in_shape must be provided upfront.
        model = _dgs_module.DeepGraphSurv(
            in_shape=(feat_dim,),
            att_dim=args.att_dim,
            hidden_dim=args.hidden_dim,
            n_layers_rep=args.n_layers_rep,
            n_layers_att=args.n_layers_att,
            K=args.K,
            dropout=args.dropout,
            compute_lambda_max=False,
        )

    # --- Wrap for diagnosis-code late fusion (matches trained checkpoint) -----
    diag_code = None
    if args.use_diagnosis:
        import json

        from late_fusion import LateFusionSurv

        with open(args.diagnosis_codes_json, encoding="utf-8") as f:
            codes_meta = json.load(f)
        n_codes = codes_meta.get("n_codes") or (max(codes_meta["codes"].values()) + 1)
        if args.diagnosis_code is not None:
            diag_code = args.diagnosis_code
        else:
            diag_code = codes_meta["codes"].get(args.diagnosis.strip(), 0)
            if diag_code == 0:
                print(
                    f"Warning: diagnosis '{args.diagnosis}' not in vocabulary — "
                    "using code 0 (unknown).",
                    file=sys.stderr,
                )
        model = LateFusionSurv(
            model, args.model_type, n_codes=n_codes, diag_dim=args.diag_dim
        )
        print(f"Late fusion: diagnosis code {diag_code} of {n_codes}.")

    # --- Load checkpoint -----------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, D)
    mask = torch.ones(1, n_patches, dtype=torch.uint8).to(device)             # (1, N)

    # --- Build adjacency (DeepGraphSurv only) --------------------------------
    adj = None
    if args.model_type == "deepgraphsurv":
        coords = load_coords(args.coords_folder)
        if len(coords) != n_patches:
            raise ValueError(
                f"Coords/features patch count mismatch: {len(coords)} vs {n_patches}."
            )
        print(f"Building adjacency (dist_thr={args.dist_thr})…")
        adj = build_dense_adj(coords, args.dist_thr).to(device)  # (1, N, N)

    # --- Inference -----------------------------------------------------------
    with torch.no_grad():
        if args.use_diagnosis:
            # The wrapper reads the code from batch["Y"][:, 2]; time/event are
            # unused at inference so are left as zeros.
            batch = {
                "X": x,
                "mask": mask,
                "Y": torch.tensor([[0.0, 0.0, float(diag_code)]], device=device),
            }
            if adj is not None:
                batch["adj"] = adj
            out = model(batch, return_att=args.return_att)
            output, att = out if args.return_att else (out, None)
        elif args.model_type == "abmil":
            if args.return_att and _TORCHMIL_AVAILABLE:
                output, att = model(x, mask, return_att=True)
            else:
                output = model(x, mask)
                att = None
        else:  # deepgraphsurv
            if args.return_att:
                output, att = model(x, adj, mask, return_att=True)
            else:
                output = model(x, adj, mask)
                att = None

    risk = output.squeeze().item()
    print(f"\nRisk score : {risk:.6f}")
    print(f"Prognosis  : {'high risk' if risk > 0 else 'low risk'} (sign-based threshold)")

    if att is not None:
        att_np = att.squeeze(0).cpu().numpy()  # (N,)
        top_k = min(10, n_patches)
        top_idx = np.argsort(att_np)[::-1][:top_k]
        print(f"\nTop-{top_k} most attended patches (index, weight):")
        for idx in top_idx:
            print(f"  patch {idx:>5d}  {att_np[idx]:.6f}")

    return risk


if __name__ == "__main__":
    main()
