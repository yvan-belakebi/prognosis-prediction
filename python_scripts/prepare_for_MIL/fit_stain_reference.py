"""
fit_stain_reference.py — Fit a Vahadane stain reference from CLAM-tiled WSIs.

Randomly samples patches from every slide in patches_dir, discards background patches
using a luminosity-based tissue mask, tiles the kept tissue patches into one large
montage image and fits a single Vahadane stain separation (torch-staintools) to it.
The result is a .pt file with keys 'stain_matrix_target' (1, 2, 3), 'maxC_target'
(1, 2) and 'method' ('vahadane') that can be passed to compute_feats_clam.py via
--stain_refs_dir.

Unlike the previous Reinhard reference (a grand mean of per-channel LAB statistics),
Vahadane fits to a *single* target image.  We therefore build a cohort-level target by
montaging tissue patches sampled across all slides of the stain; the Vahadane luminosity
mask ignores any white padding between tiles.

For datasets with multiple staining protocols, pass --labels_csv to restrict sampling to
slides of each stain, producing one reference file per stain.  Then pass the matching
--stain_refs_dir when extracting features for each stain group.

Usage:
    # Single stain
    python python_scripts/prepare_for_MIL/fit_stain_reference.py \\
        --patches_dir WSI/IgA/patches \\
        --wsi_dir     data/raw_wsi/IgA \\
        --output      stain_refs/IgA

    # One reference per stain (run once over all stains in the CSV)
    python python_scripts/prepare_for_MIL/fit_stain_reference.py \\
        --patches_dir WSI/IgA/patches \\
        --wsi_dir     data/raw_wsi/IgA \\
        --output      stain_refs/IgA \\
        --labels_csv  label_csvs/stain_labels.csv

Stain CSV format (same as used by MIL training scripts):
    file_name,stain
    slide_001,PAS
    slide_002,IgG - IgM - IgA
    ...
"""

import argparse
import math
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve sibling packages
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))

_staintools_dir = os.path.abspath(
    os.path.join(_script_dir, "..", "torch-staintools-main")
)
if _staintools_dir not in sys.path:
    sys.path.insert(0, _staintools_dir)

_clam_dir = os.path.abspath(os.path.join(_script_dir, "..", "CLAM-master"))
if _clam_dir not in sys.path:
    sys.path.insert(0, _clam_dir)

import openslide  # noqa: E402
from torch_staintools.constants import CONFIG  # noqa: E402
from torch_staintools.functional.tissue_mask import get_tissue_mask  # noqa: E402
from torch_staintools.normalizer import NormalizerBuilder  # noqa: E402

# torch.compile recompiles on each new input shape. Fitting is a one-off over a single
# montage, so compilation overhead would not be amortised — keep it off here. (It is
# enabled in compute_feats_clam.py, where the normalizer runs over many batches.)
CONFIG.ENABLE_COMPILE = False

# Concentration solver. 'qr' is robust both for the single large montage and for batched
# small tiles; 'ls' can fail on GPU for a single large image.
_CONCENTRATION_SOLVER = "qr"
_LUMINOSITY_THRESHOLD = 0.8
# Lowered solver work (factory defaults 60 / 50), matching compute_feats_clam.py so the
# reference is built with the same Vahadane estimation settings used at inference.
_SPARSE_DICT_STEPS = 30
_MAXITER = 30


def _sanitize_stain_name(stain: str) -> str:
    """Convert a stain name to a safe filename stem.

    e.g. 'IgG - IgM - IgA' → 'IgG_-_IgM_-_IgA',  'PAS' → 'PAS'
    Must match the identical function in compute_feats_clam.py.
    """
    return re.sub(r"[^\w\-]+", "_", stain).strip("_")


def _build_normalizer(method, device):
    """Build a stain-separation normalizer (torch-staintools) on the given device.

    method: 'vahadane' (sparse NMF dictionary learning) or 'macenko' (SVD + percentile).
    Both produce the same target buffers ('stain_matrix_target', 'maxC_target') and share
    the same transform; only the stain-matrix estimator differs. 'sparse_dict_steps' is a
    Vahadane-only dictionary-learning knob and is omitted for Macenko.
    """
    kwargs = dict(
        concentration_solver=_CONCENTRATION_SOLVER,
        maxiter=_MAXITER,
        luminosity_threshold=_LUMINOSITY_THRESHOLD,
        device=device,
    )
    if method == "vahadane":
        kwargs["sparse_dict_steps"] = _SPARSE_DICT_STEPS
    return NormalizerBuilder.build(method, **kwargs).to(device)


# ---------------------------------------------------------------------------
# Slide discovery  (mirrors compute_feats_clam.py)
# ---------------------------------------------------------------------------


def _discover_coord_files(patches_dir, slide_filter=None):
    """Return list of (h5_path, biopsy_nr, slide_id).

    slide_filter: optional set of bare slide_id strings (no extension) to keep.
    Supports flat (patches_dir/*.h5) and nested (patches_dir/{biopsy}/*.h5) layouts.
    """
    results = []
    has_nested = False

    for entry in sorted(os.scandir(patches_dir), key=lambda e: e.name):
        if entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file() and sub.name.endswith(".h5"):
                    slide_id = os.path.splitext(sub.name)[0]
                    bag_name = (
                        f"{entry.name}/{slide_id}"  # matches discover_bags() format
                    )
                    if slide_filter is None or bag_name in slide_filter:
                        results.append((sub.path, entry.name, slide_id))
                    has_nested = True

    if not has_nested:
        for entry in sorted(os.scandir(patches_dir), key=lambda e: e.name):
            if entry.is_file() and entry.name.endswith(".h5"):
                slide_id = os.path.splitext(entry.name)[0]
                if slide_filter is None or slide_id in slide_filter:
                    results.append((entry.path, "", slide_id))

    return results


def _find_wsi(wsi_dir, biopsy_nr, slide_id, slide_ext):
    candidates = []
    if biopsy_nr:
        candidates.append(os.path.join(wsi_dir, biopsy_nr, slide_id + slide_ext))
    candidates.append(os.path.join(wsi_dir, slide_id + slide_ext))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _build_biopsy_lookup(wsi_dir, slide_exts):
    """Map slide_id -> biopsy_nr by scanning a nested raw-WSI dir.

    TRIDENT coord files are flat (no biopsy subdir), so the biopsy_nr is recovered from the
    raw-WSI layout (mirrors reorganize_trident_feats.py / run_clam_tiling.py discovery).
    """

    def _matches(name):
        return any(name.lower().endswith(ext) for ext in slide_exts)

    lookup = {}
    has_nested = False
    for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
        if entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file() and _matches(sub.name):
                    lookup[os.path.splitext(sub.name)[0]] = entry.name
                    has_nested = True
    if not has_nested:
        for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
            if entry.is_file() and _matches(entry.name):
                lookup[os.path.splitext(entry.name)[0]] = ""
    return lookup


def _discover_trident_coord_files(patches_dir, biopsy_lookup):
    """Return list of (h5_path, biopsy_nr, slide_id) for TRIDENT coord files.

    TRIDENT writes flat '{slide_id}_patches.h5' files under '{job_dir}/{mag}x_.../patches'.
    biopsy_nr is recovered from biopsy_lookup so the output matches _discover_coord_files
    (so labels filtering and WSI lookup behave identically to the CLAM path).
    """
    results = []
    for entry in sorted(os.scandir(patches_dir), key=lambda e: e.name):
        if entry.is_file() and entry.name.endswith("_patches.h5"):
            slide_id = entry.name[: -len("_patches.h5")]
            results.append((entry.path, biopsy_lookup.get(slide_id, ""), slide_id))
    return results


def _read_coord_file(h5_path):
    """Read coords + a format-tagged read recipe from a CLAM or TRIDENT coord file.

    Both store 'coords' as (N, 2) level-0 pixel top-left corners. They differ in attrs:
        CLAM:    patch_size, patch_level          → read directly at patch_level.
        TRIDENT: patch_size, level0_magnification, target_magnification
                 → read at the best level for downsample=level0/target, then resize.
    The format is auto-detected per file, so a mixed directory is fine.
    """
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)

    if "patch_level" in attrs:
        return coords, {
            "fmt": "clam",
            "patch_size": int(attrs["patch_size"]),
            "patch_level": int(attrs["patch_level"]),
        }
    if "target_magnification" in attrs and "level0_magnification" in attrs:
        return coords, {
            "fmt": "trident",
            "patch_size": int(attrs["patch_size"]),
            "downsample": float(attrs["level0_magnification"])
            / float(attrs["target_magnification"]),
        }
    raise KeyError(
        f"Unrecognized coord attrs in {h5_path!r} (keys: {sorted(attrs)}); "
        "expected CLAM ('patch_level') or TRIDENT ('target_magnification')."
    )


def _read_patch_rgb(wsi, x, y, params):
    """Read one patch as a uint8 RGB (H, W, 3) array, matching how the encoder will see it.

    For TRIDENT, this reproduces WSIPatcher.get_tile_xy: read at the best pyramid level for
    the target downsample, then resize to patch_size (so the calibration patch resolution
    matches the feature-extraction patch resolution).
    """
    ps = params["patch_size"]
    if params["fmt"] == "clam":
        patch = wsi.read_region((x, y), params["patch_level"], (ps, ps)).convert("RGB")
        return np.array(patch)

    # TRIDENT: coords are level-0; read at best level for the target downsample, then resize.
    downsample = params["downsample"]
    patch_size_src = round(ps * downsample)
    level = wsi.get_best_level_for_downsample(downsample)
    level_ds = wsi.level_downsamples[level]
    size_level = max(1, round(patch_size_src / level_ds))
    patch = wsi.read_region((x, y), level, (size_level, size_level)).convert("RGB")
    if patch.size != (ps, ps):
        patch = patch.resize((ps, ps))
    return np.array(patch)


# ---------------------------------------------------------------------------
# Per-patch handling
# ---------------------------------------------------------------------------


def _patch_to_tile(patch_np, tile_size, min_tissue_frac):
    """Return a (3, tile_size, tile_size) float tensor in [0, 1], or None to skip.

    Skips patches whose tissue fraction (luminosity mask) is below min_tissue_frac,
    keeping background / glass out of the reference montage.
    patch_np: uint8 RGB numpy array (H, W, 3).
    """
    t = torch.from_numpy(patch_np).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    if t.shape[-2:] != (tile_size, tile_size):
        t = torch.nn.functional.interpolate(
            t, size=(tile_size, tile_size), mode="bilinear", align_corners=False
        )
    mask = get_tissue_mask(t, _LUMINOSITY_THRESHOLD, throw_error=False)
    if mask.float().mean().item() < min_tissue_frac:
        return None
    return t.squeeze(0)


def _montage(tiles):
    """Tile a list of (3, P, P) tensors into a single (1, 3, H, W) montage.

    Arranges tiles in a near-square grid; empty cells are left white (1.0), which the
    Vahadane luminosity mask ignores during fitting.
    """
    n = len(tiles)
    cols = max(int(math.ceil(math.sqrt(n))), 1)
    rows = int(math.ceil(n / cols))
    p = tiles[0].shape[-1]
    montage = torch.ones(3, rows * p, cols * p, dtype=tiles[0].dtype)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        montage[:, r * p : (r + 1) * p, c * p : (c + 1) * p] = tile
    return montage.unsqueeze(0)


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------


def compute_reference(
    coord_files,
    wsi_dir,
    slide_ext,
    method,
    n_patches_per_slide,
    n_target_patches,
    tile_size,
    min_tissue_frac,
    device,
    rng,
    n_save=0,
):
    """Sample tissue patches across slides, montage them, and fit the normalizer once.

    coord_files: list of (h5_path, biopsy_nr, slide_id) as produced by
                 _discover_coord_files (already filtered to the slides of interest).

    Returns (stain_matrix_target, maxC_target, qc_patches):
      stain_matrix_target — float32 tensor (1, 2, 3) on CPU.
      maxC_target — float32 tensor (1, 2) on CPU.
      qc_patches — list of (slide_id, tile_tensor) for the first n_save valid
                   tissue tiles encountered; empty when n_save=0.
    """
    tiles = []
    qc_patches = []
    n_slides_used = 0

    for h5_path, biopsy_nr, slide_id in tqdm(coord_files, desc="Slides"):
        if len(tiles) >= n_target_patches:
            break

        wsi_path = _find_wsi(wsi_dir, biopsy_nr, slide_id, slide_ext)
        if wsi_path is None:
            print(f"  [WARN] WSI not found for {slide_id!r}, skipping.")
            continue

        try:
            wsi = openslide.open_slide(wsi_path)
        except Exception as exc:
            print(f"  [WARN] Cannot open {wsi_path}: {exc}")
            continue

        try:
            coords, read_params = _read_coord_file(h5_path)
        except Exception as exc:
            print(f"  [WARN] Cannot read {h5_path}: {exc}")
            wsi.close()
            continue

        n = len(coords)
        sample_idx = rng.choice(n, size=min(n_patches_per_slide, n), replace=False)

        used_this_slide = False
        for i in sample_idx:
            if len(tiles) >= n_target_patches:
                break
            x, y = int(coords[i][0]), int(coords[i][1])
            try:
                patch_np = _read_patch_rgb(wsi, x, y, read_params)
            except Exception:
                continue

            tile = _patch_to_tile(patch_np, tile_size, min_tissue_frac)
            if tile is None:
                continue
            tiles.append(tile)
            used_this_slide = True
            if len(qc_patches) < n_save:
                qc_patches.append((slide_id, tile.clone()))

        wsi.close()
        if used_this_slide:
            n_slides_used += 1

    if not tiles:
        raise RuntimeError(
            "No valid tissue patches found across all slides. "
            "Check --patches_dir, --wsi_dir, and --min_tissue_frac."
        )

    montage = _montage(tiles).to(device)
    print(
        f"Fitting {method} on a montage of {len(tiles)} tissue tiles "
        f"({n_slides_used}/{len(coord_files)} slides) — montage {tuple(montage.shape)} …"
    )

    normalizer = _build_normalizer(method, device)
    normalizer.fit(montage)

    stain_matrix_target = normalizer.stain_matrix_target.detach().to(
        "cpu", torch.float32
    )
    maxC_target = normalizer.maxC_target.detach().to("cpu", torch.float32)

    print(
        f"  stain_matrix_target {tuple(stain_matrix_target.shape)}:\n{stain_matrix_target}"
    )
    print(f"  maxC_target {tuple(maxC_target.shape)}: {maxC_target}")

    return stain_matrix_target, maxC_target, qc_patches


# ---------------------------------------------------------------------------
# QC image saving
# ---------------------------------------------------------------------------


def _save_qc_images(qc_patches, stain_matrix_target, maxC_target, method, device, out_dir):
    """Save side-by-side before/after PNG images for each tile in qc_patches.

    Each image is named '{slide_id}_{i:03d}.png' and shows the original tile on the
    left, a 4-pixel white separator, and the normalised tile on the right.
    """
    from PIL import Image

    normalizer = _build_normalizer(method, device)
    normalizer.register_buffer("stain_matrix_target", stain_matrix_target.to(device))
    normalizer.register_buffer("maxC_target", maxC_target.to(device))

    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    def _to_uint8(tile_tensor):
        arr = tile_tensor.clamp(0, 1).mul(255).round().byte()
        return arr.permute(1, 2, 0).cpu().numpy()

    for i, (slide_id, tile) in enumerate(qc_patches):
        try:
            with torch.inference_mode():
                normalized = normalizer(tile.unsqueeze(0).to(device)).squeeze(0)
        except Exception as exc:
            print(f"  [WARN] QC normalisation failed for tile {i} ({slide_id}): {exc}")
            continue

        before = _to_uint8(tile)
        after = _to_uint8(normalized)
        h = before.shape[0]
        separator = np.full((h, 4, 3), 255, dtype=np.uint8)
        side_by_side = np.concatenate([before, separator, after], axis=1)

        safe_slide_id = slide_id.replace(os.sep, "_").replace("/", "_")
        fname = os.path.join(out_dir, f"{safe_slide_id}_{i:03d}.png")
        Image.fromarray(side_by_side).save(fname)
        saved += 1

    print(f"QC images saved → {out_dir}  ({saved} before/after pairs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fit a Vahadane/Macenko stain reference from CLAM-tiled WSI patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patches_dir",
        required=True,
        help=(
            "Directory of patch-coordinate .h5 files. CLAM: 'WSI/IgA/patches' "
            "({slide}.h5, flat or biopsy-nested). TRIDENT: the 'patches' subdir of a job, "
            "e.g. 'WSI/IgA/trident/20x_224px_0px_overlap/patches' ({slide}_patches.h5)."
        ),
    )
    parser.add_argument(
        "--coords_format",
        choices=["auto", "clam", "trident"],
        default="auto",
        help=(
            "Coordinate-file layout. 'auto' picks TRIDENT when the dir holds "
            "'*_patches.h5' files, else CLAM. The per-file read recipe is always "
            "auto-detected from the h5 attributes regardless of this flag."
        ),
    )
    parser.add_argument(
        "--wsi_dir",
        required=True,
        help="Root directory of raw WSI files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help=(
            "Output directory for .pt reference files. "
            "Without --labels_csv: saves a single 'reference.pt'. "
            "With --labels_csv: saves one '{stain}.pt' per stain."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["vahadane", "macenko"],
        default="vahadane",
        help=(
            "Stain-separation algorithm. 'vahadane' (sparse NMF) gives better stain "
            "fidelity; 'macenko' (SVD + percentile) is much faster. The chosen method is "
            "recorded in the .pt and auto-matched by compute_feats_clam.py."
        ),
    )
    parser.add_argument(
        "--slide_ext",
        default=".svs",
        help="WSI file extension.",
    )
    parser.add_argument(
        "--n_patches_per_slide",
        type=int,
        default=50,
        help="Number of patches randomly sampled per slide.",
    )
    parser.add_argument(
        "--n_target_patches",
        type=int,
        default=100,
        help=(
            "Number of tissue tiles to collect into the reference montage. "
            "Sampling stops once this many valid tiles are gathered. "
            "Larger montages give a more representative reference but cost more "
            "GPU/CPU memory and time to fit."
        ),
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Side length each sampled patch is resized to before montaging.",
    )
    parser.add_argument(
        "--min_tissue_frac",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of tissue pixels (luminosity < 0.8) for a patch to be "
            "kept for the reference montage. Filters out background / glass."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible patch sampling.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="CUDA device index (used if a GPU is available; otherwise CPU).",
    )
    parser.add_argument(
        "--labels_csv",
        default=None,
        help=(
            "CSV with 'file_name' and 'stain' columns (same format as MIL training scripts). "
            "When provided, one reference .pt is computed and saved per unique stain. "
            "Omit to compute a single reference for all slides."
        ),
    )
    parser.add_argument(
        "--save_patches",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write before/after normalisation QC images. "
            "Each image shows a sampled tissue tile before (left) and after (right) "
            "Vahadane normalisation. Omit to skip QC image saving."
        ),
    )
    parser.add_argument(
        "--n_save_patches",
        type=int,
        default=5,
        help="Number of patches to save when --save_patches is set.",
    )

    args = parser.parse_args()

    slide_ext = args.slide_ext
    if not slide_ext.startswith("."):
        slide_ext = f".{slide_ext}"

    device = torch.device(
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}  |  method: {args.method}")

    os.makedirs(args.output, exist_ok=True)
    n_save = args.n_save_patches if args.save_patches is not None else 0

    # Resolve the coord-file layout. The per-file read recipe is detected separately from
    # the h5 attrs, so 'auto' here only decides discovery (flat *_patches.h5 vs CLAM *.h5).
    fmt = args.coords_format
    if fmt == "auto":
        has_trident = any(
            e.is_file() and e.name.endswith("_patches.h5")
            for e in os.scandir(args.patches_dir)
        )
        fmt = "trident" if has_trident else "clam"

    # Scan the filesystem once; a genuinely empty patches_dir is fatal, but
    # individual stains with no patched slides are skipped below (not fatal).
    if fmt == "trident":
        biopsy_lookup = _build_biopsy_lookup(args.wsi_dir, (slide_ext,))
        all_coord_files = _discover_trident_coord_files(args.patches_dir, biopsy_lookup)
    else:
        all_coord_files = _discover_coord_files(args.patches_dir)
    if not all_coord_files:
        raise RuntimeError(
            f"No {'*_patches.h5' if fmt == 'trident' else '*.h5'} coord files "
            f"found in {args.patches_dir!r} (format: {fmt})."
        )

    def _bag_name(entry):
        _, biopsy_nr, slide_id = entry
        return f"{biopsy_nr}/{slide_id}" if biopsy_nr else slide_id

    def _run_one(stain_label, slide_filter):
        """Compute and save a single reference; return the output path, or None if skipped."""
        if slide_filter is None:
            coord_files = all_coord_files
        else:
            coord_files = [e for e in all_coord_files if _bag_name(e) in slide_filter]

        if stain_label is not None:
            stem = _sanitize_stain_name(stain_label)
            out_path = os.path.join(args.output, f"{stem}.pt")
            qc_dir = (
                os.path.join(args.save_patches, stem) if args.save_patches else None
            )
            print(
                f"\nStain '{stain_label}' "
                f"({len(slide_filter)} in CSV, {len(coord_files)} patched) …"
            )
            if not coord_files:
                print(f"  [SKIP] No patched slides for stain {stain_label!r}.")
                return None
        else:
            out_path = os.path.join(args.output, "reference.pt")
            qc_dir = args.save_patches

        rng = np.random.default_rng(args.seed)  # same seed → reproducible per stain
        stain_matrix_target, maxC_target, qc_patches = compute_reference(
            coord_files=coord_files,
            wsi_dir=args.wsi_dir,
            slide_ext=slide_ext,
            method=args.method,
            n_patches_per_slide=args.n_patches_per_slide,
            n_target_patches=args.n_target_patches,
            tile_size=args.tile_size,
            min_tissue_frac=args.min_tissue_frac,
            device=device,
            rng=rng,
            n_save=n_save,
        )
        if qc_dir is not None:
            _save_qc_images(
                qc_patches, stain_matrix_target, maxC_target, args.method, device, qc_dir
            )
        torch.save(
            {
                "stain_matrix_target": stain_matrix_target,
                "maxC_target": maxC_target,
                "method": args.method,
            },
            out_path,
        )
        print(f"Reference saved → {out_path}")
        return out_path

    if args.labels_csv is not None:
        import pandas as pd

        df = pd.read_csv(args.labels_csv)
        stains = sorted(df["stain"].dropna().unique())
        print(f"Found {len(stains)} stain(s) in {args.labels_csv}: {stains}")
        for stain in stains:
            slide_filter = set(df.loc[df["stain"] == stain, "file_name"].astype(str))
            _run_one(stain, slide_filter)
    else:
        _run_one(None, None)


if __name__ == "__main__":
    main()
