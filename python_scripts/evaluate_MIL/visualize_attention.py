"""
visualize_attention.py — Select prognosis-stratified biopsies and visualize attention maps.

Picks one biopsy from each of three strata:
  bad    : event occurred, time in the bottom --bad_pct percentile of event times.
  medium : event occurred, time between --bad_pct and --good_pct percentile.
  good   : censored (event==0); falls back to the longest event times if no censored cases.

For each selected biopsy the script runs a trained MIL model with return_att=True,
normalises the per-patch attention weights, and overlays them as a colour heatmap
(green = low, red = high) on a spatial canvas reconstructed from patch coordinates.

Usage:
    # Survival ABMIL — coords auto-read from the features .h5 (no --coords_paths needed)
    python python_scripts/MIL/visualize_attention.py \\
        --label_csv      results/eval/risk_scores.csv \\
        --features_paths WSI/IgA/UNI2-h_feats \\
        --checkpoint     runs/abmil_survival/best_model.pth \\
        --model_type     abmil --task survival \\
        --patch_size     256

    # Regression DSMIL, two datasets — coords embedded in features .h5
    python python_scripts/MIL/visualize_attention.py \\
        --label_csv      label_csvs/labels.csv \\
        --features_paths WSI/IgA/UNI2-h_feats WSI/IgA_registry/UNI2-h_feats \\
        --checkpoint     runs/dsmil_reg/best_model.pth \\
        --model_type     dsmil --task regression \\
        --patch_size     256

    # Survival DeepGraphSurv — coords auto-read from features .h5 for adjacency
    python python_scripts/MIL/visualize_attention.py \\
        --label_csv      results/eval/risk_scores.csv \\
        --features_paths WSI/IgA/UNI2-h_feats \\
        --checkpoint     runs/dgs_survival/best_model.pth \\
        --model_type     deepgraphsurv --task survival \\
        --patch_size     256 --dist_thr 1.5 --n_layers_rep 1 --n_layers_att 1 --K 5

The --label_csv can be any CSV that has biopsy/bag_name/file_name/id, time, and event
columns — including the risk_scores.csv produced by evaluate_survival.py.
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Resolve the local torchmil package
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_torchmil_root = os.path.join(_project_root, "torchmil")
if (
    os.path.isdir(os.path.join(_torchmil_root, "torchmil"))
    and _torchmil_root not in sys.path
):
    sys.path.insert(0, _torchmil_root)

from torchmil.models import abmil as abmil_module
from torchmil.models import deepgraphsurv as dgs_module
from torchmil.models import dsmil as dsmil_module
from torchmil.models import transmil as transmil_module
from torchmil.utils import add_self_loops, build_adj, normalize_adj
from torchmil.visualize.vis_wsi import draw_heatmap_wsi

# Deep blue (high attention) → Pink (low attention)
_HIGH_COLOR = np.array([0.039, 0.118, 0.588])
_LOW_COLOR = np.array([1.0, 0.588, 0.706])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_label_csv(path: str) -> pd.DataFrame:
    """Load label CSV, normalise the ID column to 'biopsy', coerce time/event."""
    df = pd.read_csv(path)
    for candidate in ("biopsy", "bag_name", "file_name", "id"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "biopsy"})
            break
    else:
        raise ValueError(
            f"CSV must contain a biopsy/bag_name/file_name/id column; "
            f"found: {list(df.columns)}"
        )
    for col in ("time", "event"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column.")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype("Int64")
    return df.dropna(subset=["biopsy", "time", "event"]).reset_index(drop=True)


def _find_bag_file(base_paths: list, bag_name: str) -> tuple[str | None, str]:
    """Search base_paths for bag_name, returning (file_path, resolved_bag_name).

    Supports three layouts:
      flat   : base_path/slide.h5           bag_name = "slide"
      nested : base_path/biopsy/slide.h5    bag_name = "biopsy/slide"
      biopsy : base_path/biopsy/slide.h5    bag_name = "biopsy"  (directory match:
                                             first slide inside is selected)

    resolved_bag_name equals bag_name for the first two layouts.  For the third it
    becomes "biopsy/slide" so that the coords directory can be searched with the
    same specific slide path.
    """
    for base_path in base_paths:
        # Direct file: flat ("slide") or already-nested ("biopsy/slide")
        for ext in (".h5", ".npy"):
            p = os.path.join(base_path, bag_name + ext)
            if os.path.isfile(p):
                return p, bag_name
        # Biopsy-level: bag_name is a directory — pick the first slide inside
        biopsy_dir = os.path.join(base_path, bag_name)
        if os.path.isdir(biopsy_dir):
            for entry in sorted(os.scandir(biopsy_dir), key=lambda e: e.name):
                if entry.is_file() and os.path.splitext(entry.name)[1] in (
                    ".h5",
                    ".npy",
                ):
                    slide_stem = os.path.splitext(entry.name)[0]
                    resolved = f"{bag_name}/{slide_stem}"
                    return entry.path, resolved
    return None, bag_name


def _load_array(file_path: str, h5_key: str) -> np.ndarray:
    if file_path.endswith(".h5"):
        with h5py.File(file_path, "r") as f:
            return f[h5_key][:]
    return np.load(file_path)


def load_bag(bag_name: str, features_paths: list, coords_paths: list | None):
    """Return (X, coords) where X is (N, D) and coords is (N, 2) or None.

    Searches each directory in features_paths / coords_paths in order.
    Supports flat, biopsy-nested ("biopsy/slide"), and biopsy-level ("biopsy")
    bag names — see _find_bag_file for resolution rules.

    Coord loading priority:
      1. "coords" key inside the features .h5 file — guaranteed to be aligned with
         features because compute_feats_clam.py writes both keys from the same
         CLAM coordinate file.
      2. Separate coords file from coords_paths — legacy fallback; prints a warning
         when patch counts differ.
    """
    feat_file, resolved = _find_bag_file(features_paths, bag_name)
    if feat_file is None:
        searched = ", ".join(features_paths)
        raise FileNotFoundError(
            f"No feature file found for '{bag_name}' in [{searched}]"
        )
    if resolved != bag_name:
        print(f"  Biopsy-level ID '{bag_name}' → slide '{resolved}'")

    # Load features; also check for an aligned coords key in the same h5 file
    coords = None
    if feat_file.endswith(".h5"):
        with h5py.File(feat_file, "r") as f:
            X = f["features"][:]
            if "coords" in f:
                coords = f["coords"][:]
    else:
        X = np.load(feat_file)

    if X.ndim == 1:
        X = X[np.newaxis, :]

    # Fall back to separate coords file when h5 did not contain coords
    if coords is None and coords_paths:
        coord_file, _ = _find_bag_file(coords_paths, resolved)
        if coord_file is not None:
            coords = _load_array(coord_file, "coords")
            n_feats, n_coords = len(X), len(coords)
            if n_feats != n_coords:
                print(
                    f"  Warning: features has {n_feats} patches but coords has "
                    f"{n_coords}.  Patch counts differ — spatial assignment of "
                    f"attention weights may be wrong.  Consider using features "
                    f".h5 files that contain an embedded 'coords' key (produced "
                    f"by compute_feats_clam.py) to guarantee alignment."
                )

    return X, coords


# ---------------------------------------------------------------------------
# Prognosis stratification
# ---------------------------------------------------------------------------


def select_biopsies(df: pd.DataFrame, bad_pct: float, good_pct: float, seed: int):
    """Return {'bad': row, 'medium': row, 'good': row} with one sampled biopsy each.

    bad    : event==1 AND time <= bad_pct-th percentile of event times
    medium : event==1 AND bad_pct < time <= good_pct-th percentile
    good   : event==0 (censored); falls back to event==1 with time > good_pct cutoff
    """
    event_times = df.loc[df["event"] == 1, "time"]
    if event_times.empty:
        raise ValueError("No events (event==1) in the label CSV — cannot stratify.")

    bad_cut = float(np.percentile(event_times, bad_pct))
    good_cut = float(np.percentile(event_times, good_pct))

    pools = {
        "bad": df[(df["event"] == 1) & (df["time"] <= bad_cut)],
        "medium": df[
            (df["event"] == 1)
            & (df["time"] > bad_cut)
            & (df["time"] <= (good_cut + bad_cut) / 2)
        ],
        "good": df[
            (df["event"] == 0) & (df["time"] >= good_cut)
        ],  # prefer censored with long follow-up
    }
    # Fallback for good: use longest event times if no censored cases
    if pools["good"].empty:
        pools["good"] = df[(df["event"] == 1) & (df["time"] >= good_cut)]

    selected = {}
    for name, pool in pools.items():
        if pool.empty:
            print(f"  Warning: empty '{name}' pool — skipping.")
            selected[name] = None
        else:
            row = pool.sample(1, random_state=seed).iloc[0]
            selected[name] = row
            print(
                f"  {name:6s}: {row['biopsy']}  "
                f"time={row['time']:.1f}  event={int(row['event'])}"
            )
    return selected


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(model_type: str, task: str, feat_dim: int, args) -> nn.Module:
    """Construct the model with the same architecture used during training."""
    if task == "survival":
        if model_type == "abmil":
            # MIL.py uses lazy input (no in_shape)
            model = abmil_module.ABMIL(att_dim=args.att_dim)
        elif model_type == "dsmil":
            model = dsmil_module.DSMIL(
                in_shape=(feat_dim,),
                att_dim=args.att_dim,
                nonlinear_q=args.nonlinear_q,
                nonlinear_v=args.nonlinear_v,
                dropout=0.0,
            )
        elif model_type == "transmil":
            model = transmil_module.TransMIL(
                in_shape=(feat_dim,),
                att_dim=args.att_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dropout=0.0,
            )
        elif model_type == "deepgraphsurv":
            model = dgs_module.DeepGraphSurv(
                in_shape=(feat_dim,),
                att_dim=args.att_dim,
                hidden_dim=args.hidden_dim,
                n_layers_rep=args.n_layers_rep,
                n_layers_att=args.n_layers_att,
                K=args.K,
                dropout=0.0,
                compute_lambda_max=False,
            )
        else:
            raise ValueError(f"model_type '{model_type}' not supported for survival.")
    else:  # regression
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
                dropout=0.0,
            )
        elif model_type == "transmil":
            model = transmil_module.TransMIL(
                in_shape=(feat_dim,),
                att_dim=args.att_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                dropout=0.0,
            )
        else:
            raise ValueError(f"model_type '{model_type}' not supported for regression.")
        # Match regression_MIL.py: replace head with single-output linear
        model.classifier = nn.LazyLinear(1)

    # Materialise any LazyLinear layers before loading the checkpoint
    dummy_x = torch.zeros(1, 1, feat_dim)
    dummy_mask = torch.ones(1, 1, dtype=torch.uint8)
    with torch.no_grad():
        if model_type == "deepgraphsurv":
            dummy_adj = torch.ones(1, 1, 1)
            model(dummy_x, dummy_adj, dummy_mask)
        elif model_type == "transmil":
            model(dummy_x)
        else:
            model(dummy_x, dummy_mask)

    return model


def load_checkpoint(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support checkpoints that wrap state_dict in a dict
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _build_adj(coords: np.ndarray, dist_thr: float) -> torch.Tensor:
    """Build a normalised dense adjacency matrix from patch coordinates (N, 2).

    Returns a tensor of shape (1, N, N), matching DeepGraphSurv's expected input.

    When coords are in pixel space (as produced by patches_to_coords.py), dist_thr
    must be in the same unit — e.g. 1.5 * tile_size (~336 for 224 px tiles, ~384
    for 256 px tiles).  Use the same value that was set during training.
    """
    if coords.max() > 500 and dist_thr < 10:
        print(
            f"  Warning: coords appear to be in pixel space (max={coords.max():.0f}) "
            f"but --dist_thr={dist_thr} is very small.  The adjacency graph will be "
            f"empty (no neighbours connected).  Set --dist_thr to the value used "
            f"during training — typically 1.5 × tile_size "
            f"(e.g. {1.5 * 224:.0f} for 224 px tiles, {1.5 * 256:.0f} for 256 px)."
        )
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


def run_inference(
    model: nn.Module,
    model_type: str,
    X: np.ndarray,
    device: torch.device,
    coords: np.ndarray | None = None,
    dist_thr: float = 1.5,
):
    """Return (risk_score: float, att_weights: ndarray of shape (N,))."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1,N,D)
    mask_t = torch.ones(1, X.shape[0], dtype=torch.uint8).to(device)  # (1,N)

    model.eval()
    with torch.no_grad():
        if model_type == "deepgraphsurv":
            if coords is None:
                raise ValueError(
                    "deepgraphsurv requires patch coordinates.  "
                    "Ensure the features .h5 contains an embedded 'coords' key "
                    "(produced by compute_feats_clam.py), or supply --coords_paths."
                )
            adj = _build_adj(coords, dist_thr).to(device)  # (1, N, N)
            pred, att = model(X_t, adj, mask_t, return_att=True)
            att_np = att.squeeze(0).cpu().numpy()  # (N,)
        elif model_type == "transmil":
            pred, att = model(X_t, return_att=True)
            att_np = att.squeeze(0).cpu().numpy()  # (N,)
        elif model_type == "dsmil":
            pred, att = model(X_t, mask_t, return_att=True)
            att_np = att.squeeze(0).squeeze(-1).cpu().numpy()  # (N,1) → (N,)
        else:  # abmil
            pred, att = model(X_t, mask_t, return_att=True)
            # ABMIL returns raw logits — convert to probability-like weights
            att_np = torch.softmax(att.squeeze(0), dim=0).cpu().numpy()  # (N,)

    risk = pred.squeeze().item()
    return risk, att_np


# ---------------------------------------------------------------------------
# Canvas construction
# ---------------------------------------------------------------------------


def _coords_to_grid(coords: np.ndarray, coords_in_pixels: bool, patch_size: int):
    """Convert (N,2) coords to integer (row_array, col_array) grid indices.

    Pixel coordinates (as produced by patches_to_coords.py) are detected
    automatically when their maximum value exceeds 500 — a threshold safely above
    any realistic grid index count.  Pass --patch_size matching the tile size
    used during patch extraction (default 256; patches_to_coords.py defaults to 224).
    """
    if coords_in_pixels or coords.max() > 500:
        if not coords_in_pixels:
            print(
                f"  Auto-detected pixel coordinates (max={coords.max():.0f}); "
                f"dividing by --patch_size={patch_size} to get grid indices."
            )
        row_array = (coords[:, 1] // patch_size).astype(int)
        col_array = (coords[:, 0] // patch_size).astype(int)
    else:
        row_array = coords[:, 1].astype(int)
        col_array = coords[:, 0].astype(int)
    return row_array, col_array


def make_attention_canvas(
    coords: np.ndarray,
    att_weights: np.ndarray,
    display_cell_size: int,
    coords_in_pixels: bool,
    patch_size: int,
    alpha: float,
    att_percentile: float = 100.0,
) -> np.ndarray:
    """Build an RGB canvas with a spatial attention heatmap.

    Each patch position is painted light grey (tissue mask), then the attention
    heatmap is blended on top.  display_cell_size controls how many pixels each
    grid cell occupies in the output image.

    att_percentile controls how the colour scale is set:
      100 (default) — full min/max range.  Skewed distributions (a few very high
                       patches, many near-zero) will make most of the map green.
      95 or 99      — clip to [p, 100-p] percentile before normalising.  Spreads
                       the colour scale over the bulk of the distribution and avoids
                       the "everything is 0 except a few patches" artefact.

    When coords has more rows than att_weights (patch-count mismatch between the
    coords file and the feature extractor), only the first len(att_weights) coord
    rows are coloured; the remainder stay grey.  This is imperfect — see load_bag
    for the recommended fix.
    """
    row_array, col_array = _coords_to_grid(coords, coords_in_pixels, patch_size)

    n_coords = len(row_array)
    n_att = len(att_weights)

    if n_coords != n_att:
        # Colour only the patches we have attention weights for; the rest stay grey.
        # This is a best-effort fallback — spatial positions may not be correctly
        # assigned when the features file is a filtered subset of the coords file.
        n_use = min(n_coords, n_att)
        row_array = row_array[:n_use]
        col_array = col_array[:n_use]
        att_weights = att_weights[:n_use]

    H = (row_array.max() + 1) * display_cell_size
    W = (col_array.max() + 1) * display_cell_size

    # White background + light-grey tissue mask for ALL coord positions
    all_rows, all_cols = _coords_to_grid(coords, coords_in_pixels, patch_size)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for r, c in zip(all_rows, all_cols):
        canvas[
            r * display_cell_size : (r + 1) * display_cell_size,
            c * display_cell_size : (c + 1) * display_cell_size,
        ] = 210

    # Normalise attention to [0, 1] with optional percentile clipping
    att = att_weights.astype(np.float64)
    if att_percentile < 100.0:
        lo = np.percentile(att, 100.0 - att_percentile)
        hi = np.percentile(att, att_percentile)
        att = np.clip(att, lo, hi)
        vmin, vmax = lo, hi
    else:
        vmin, vmax = att.min(), att.max()
    att = (att - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(att)

    canvas = draw_heatmap_wsi(
        canvas,
        att,
        display_cell_size,
        row_array,
        col_array,
        alpha=alpha,
        max_color=_HIGH_COLOR,
        min_color=_LOW_COLOR,
    )
    return canvas


def save_attention_tif(
    coords: np.ndarray,
    att_weights: np.ndarray,
    category: str,
    biopsy_short: str,
    output_dir: str,
    coords_in_pixels: bool,
    patch_size: int,
    alpha: float,
    att_percentile: float,
    blur_sigma: float,
) -> None:
    """Save a full-resolution TIFF of the attention map.

    Unlike the PNG figures (which use --display_cell_size pixels per patch), the
    TIFF renders each patch as a patch_size × patch_size square so the output
    matches the original spatial scale of the extracted objects.  A Gaussian blur
    of the given sigma (px) is applied to smooth the blocky per-patch appearance.
    """
    canvas = make_attention_canvas(
        coords,
        att_weights,
        display_cell_size=patch_size,
        coords_in_pixels=coords_in_pixels,
        patch_size=patch_size,
        alpha=alpha,
        att_percentile=att_percentile,
    )
    img = Image.fromarray(canvas)
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    safe_name = biopsy_short.replace("/", "_").replace("\\", "_")
    tif_path = os.path.join(output_dir, f"attention_{category}_{safe_name}.tif")
    img.save(tif_path, format="TIFF")
    print(f"  Saved TIF → {tif_path}  ({img.width}×{img.height}px, blur σ={blur_sigma:g})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STRATUM_LABELS = {
    "bad": "Bad prognosis\n(short time to event)",
    "medium": "Medium prognosis\n(long time to event)",
    "good": "Good prognosis\n(censored or long follow-up)",
}


def plot_results(results: dict, model_type: str, task: str, output_dir: str):
    active = [(cat, res) for cat, res in results.items() if res is not None]
    n = len(active)
    if n == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (category, res) in zip(axes, active):
        row = res["row"]
        biopsy_short = os.path.basename(str(row["biopsy"]))

        if res["canvas"] is not None:
            ax.imshow(res["canvas"])
        else:
            ax.text(
                0.5, 0.5, "No coords", ha="center", va="center", transform=ax.transAxes
            )

        ax.set_title(
            f"{_STRATUM_LABELS[category]}\n"
            f"{biopsy_short}\n"
            f"time = {row['time']:.1f}   event = {int(row['event'])}   "
            f"risk = {res['risk']:.4f}",
            fontsize=9,
        )
        ax.axis("off")

        # Save individual image
        if res["canvas"] is not None:
            safe_name = biopsy_short.replace("/", "_").replace("\\", "_")
            ind_path = os.path.join(output_dir, f"attention_{category}_{safe_name}.png")
            fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
            ax_ind.imshow(res["canvas"])
            ax_ind.set_title(ax.get_title(), fontsize=8)
            ax_ind.axis("off")
            fig_ind.tight_layout()
            fig_ind.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_ind)
            print(f"  Saved individual → {ind_path}")

    # Shared colourbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("att", [_LOW_COLOR, _HIGH_COLOR])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.03, shrink=0.8)
    cbar.set_label("Attention (normalised)", fontsize=9)
    cbar.ax.set_yticks([0, 1])
    cbar.ax.set_yticklabels(["Low", "High"], fontsize=8)

    fig.suptitle(f"MIL Attention Maps — {model_type.upper()} ({task})", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"attention_comparison_{model_type}_{task}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved combined → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Attention-map visualisation for bad / medium / good prognosis biopsies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    parser.add_argument(
        "--label_csv",
        required=True,
        help="CSV with biopsy/bag_name, time, event columns.",
    )
    parser.add_argument(
        "--features_paths",
        nargs="+",
        required=True,
        help="One or more directories with bag feature files (.h5 or .npy). "
        "Biopsies are looked up in each directory in order.",
    )
    parser.add_argument(
        "--coords_paths",
        nargs="+",
        default=None,
        help="One or more directories with patch coordinate files (.h5 or .npy), "
        "in the same order as --features_paths.  "
        "Optional when features are CLAM-style .h5 files that already contain "
        "an embedded 'coords' key (produced by compute_feats_clam.py).",
    )
    parser.add_argument(
        "--output_dir",
        default="attention_maps",
        help="Directory to save output PNG figures.",
    )

    # --- Model ---
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)."
    )
    parser.add_argument(
        "--model_type",
        default="abmil",
        choices=["abmil", "dsmil", "transmil", "deepgraphsurv"],
    )
    parser.add_argument(
        "--task", default="survival", choices=["survival", "regression"]
    )

    # --- Model hyperparams (must match training) ---
    parser.add_argument("--att_dim", type=int, default=128)
    parser.add_argument(
        "--gated", action="store_true", help="Gated attention (ABMIL regression only)."
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Attention heads (TransMIL)."
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Transformer layers (TransMIL)."
    )
    parser.add_argument(
        "--nonlinear_q",
        action="store_true",
        help="Non-linear query projection (DSMIL).",
    )
    parser.add_argument(
        "--nonlinear_v",
        action="store_true",
        help="Non-linear value projection (DSMIL).",
    )
    # DeepGraphSurv-specific
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden GCN dimension (DeepGraphSurv; default: feat_dim).",
    )
    parser.add_argument(
        "--n_layers_rep",
        type=int,
        default=1,
        help="Representation GCN layers (DeepGraphSurv).",
    )
    parser.add_argument(
        "--n_layers_att",
        type=int,
        default=1,
        help="Attention GCN layers (DeepGraphSurv).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Chebyshev polynomial order (DeepGraphSurv).",
    )
    parser.add_argument(
        "--dist_thr",
        type=float,
        default=1.5,
        help="Adjacency distance threshold (DeepGraphSurv).",
    )

    # --- Stratification ---
    parser.add_argument(
        "--bad_pct",
        type=float,
        default=25.0,
        help="Percentile of event times below which prognosis is 'bad'.",
    )
    parser.add_argument(
        "--good_pct",
        type=float,
        default=75.0,
        help="Percentile of event times above which prognosis is 'good' "
        "(only applies to event==1 fallback; censored patients are "
        "always preferred for the good category).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for biopsy sampling."
    )

    # --- Visualisation ---
    parser.add_argument(
        "--display_cell_size",
        type=int,
        default=32,
        help="Pixels per grid cell in the output image.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Extraction patch size in pixels; only used when "
        "--coords_in_pixels is set.",
    )
    parser.add_argument(
        "--coords_in_pixels",
        action="store_true",
        help="Divide raw pixel coordinates by --patch_size to get "
        "grid indices.  Leave unset if coords are already grid indices.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Opacity of the attention overlay (0=transparent, 1=opaque).",
    )
    parser.add_argument(
        "--att_percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile used to set the colour scale: the heatmap spans "
            "[100-p, p] percentile of the attention distribution before "
            "normalising to [0, 1].  Values above/below are clipped.  "
            "100 = full min/max (no clipping).  95 or 99 avoids the "
            "'everything green except a few patches' artefact caused by "
            "outlier high-attention patches compressing the rest of the scale."
        ),
    )
    parser.add_argument(
        "--save_as_tif",
        action="store_true",
        help="In addition to the PNG, save a full-resolution TIFF per biopsy in "
        "which each patch is rendered as a --patch_size × --patch_size square "
        "(matching the original object scale).  A Gaussian blur "
        "(--tif_blur_sigma) is applied to smooth the per-patch blocks.",
    )
    parser.add_argument(
        "--tif_blur_sigma",
        type=float,
        default=None,
        help="Gaussian blur radius (px) applied to the TIFF output.  "
        "Default: patch_size / 4.  Set to 0 to disable blurring.",
    )

    args = parser.parse_args()

    if args.tif_blur_sigma is None:
        args.tif_blur_sigma = args.patch_size / 4.0

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Labels
    print("\nLoading labels …")
    label_df = load_label_csv(args.label_csv)
    n_events = int(label_df["event"].sum())
    print(
        f"  {len(label_df)} biopsies total,  {n_events} events,  "
        f"{len(label_df) - n_events} censored."
    )

    # 2. Select one biopsy per stratum
    print("\nSelecting biopsies:")
    selected = select_biopsies(label_df, args.bad_pct, args.good_pct, args.seed)

    # 3. Infer feat_dim from the first available biopsy
    first_row = next(v for v in selected.values() if v is not None)
    X_tmp, _ = load_bag(first_row["biopsy"], args.features_paths, coords_paths=None)
    feat_dim = X_tmp.shape[-1]
    print(f"\nFeature dimension: {feat_dim}")

    # 4. Build and load model
    print(f"Building {args.model_type} ({args.task}) model …")
    model = build_model(args.model_type, args.task, feat_dim, args)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # 5. Per-biopsy inference + canvas
    results = {}
    for category in ("bad", "medium", "good"):
        row = selected[category]
        if row is None:
            results[category] = None
            continue

        biopsy_name = row["biopsy"]
        print(f"\n[{category}] {biopsy_name}")

        try:
            X, coords = load_bag(biopsy_name, args.features_paths, args.coords_paths)
        except FileNotFoundError as e:
            print(f"  Skipping — {e}")
            results[category] = None
            continue

        risk, att = run_inference(
            model,
            args.model_type,
            X,
            device,
            coords=coords,
            dist_thr=args.dist_thr,
        )
        print(
            f"  patches={len(X)}  risk={risk:.4f}  "
            f"att=[{att.min():.4f}, {att.max():.4f}]"
        )

        canvas = None
        if coords is not None:
            canvas = make_attention_canvas(
                coords,
                att,
                display_cell_size=args.display_cell_size,
                coords_in_pixels=args.coords_in_pixels,
                patch_size=args.patch_size,
                alpha=args.alpha,
                att_percentile=args.att_percentile,
            )
            if args.save_as_tif:
                save_attention_tif(
                    coords,
                    att,
                    category,
                    os.path.basename(str(biopsy_name)),
                    args.output_dir,
                    coords_in_pixels=args.coords_in_pixels,
                    patch_size=args.patch_size,
                    alpha=args.alpha,
                    att_percentile=args.att_percentile,
                    blur_sigma=args.tif_blur_sigma,
                )
        else:
            print("  No coordinate file found — spatial map skipped.")

        results[category] = {"row": row, "risk": risk, "att": att, "canvas": canvas}

    # 6. Plot and save
    print()
    plot_results(results, args.model_type, args.task, args.output_dir)


if __name__ == "__main__":
    main()
