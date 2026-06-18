"""
fit_stain_reference.py — Compute a mean Modified Reinhard stain reference from CLAM-tiled WSIs.

Randomly samples patches from every slide in patches_dir, discards background patches using
an L* threshold, and computes the grand mean of per-channel LAB statistics across all valid
tissue patches.  The result is a .pt file with keys 'target_means' and 'target_stds' (shape 3,
float32, L/a/b order) that can be passed to compute_feats_clam.py via --stain_ref.

For datasets with multiple staining protocols, run once per stain using --stain_csv / --stain
to restrict sampling to slides of that stain, producing one reference file per stain.  Then
pass the matching reference when extracting features for each stain group.

Usage:
    # Single stain
    python python_scripts/prepare_for_MIL/fit_stain_reference.py \\
        --patches_dir WSI/IgA/patches \\
        --wsi_dir     data/raw_wsi/IgA \\
        --output      stain_refs/IgA_ref.pt

    # One of several stains (run once per stain)
    python python_scripts/prepare_for_MIL/fit_stain_reference.py \\
        --patches_dir WSI/IgA/patches \\
        --wsi_dir     data/raw_wsi/IgA \\
        --output      stain_refs/IgA_PAS_ref.pt \\
        --stain_csv   followup_data/stain_labels.csv \\
        --stain       PAS

Stain CSV format (same as used by MIL training scripts):
    file_name,stain
    slide_001,PAS
    slide_002,IgG - IgM - IgA
    ...
"""

import argparse
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

_torchstain_dir = os.path.abspath(os.path.join(_script_dir, "..", "torchstain-main"))
if _torchstain_dir not in sys.path:
    sys.path.insert(0, _torchstain_dir)

_clam_dir = os.path.abspath(os.path.join(_script_dir, "..", "CLAM-master"))
if _clam_dir not in sys.path:
    sys.path.insert(0, _clam_dir)

import openslide  # noqa: E402
from torchstain.numpy.utils.rgb2lab import rgb2lab  # noqa: E402
from torchstain.numpy.utils.split import lab_split  # noqa: E402
from torchstain.numpy.utils.stats import get_mean_std  # noqa: E402


def _sanitize_stain_name(stain: str) -> str:
    """Convert a stain name to a safe filename stem.

    e.g. 'IgG - IgM - IgA' → 'IgG_-_IgM_-_IgA',  'PAS' → 'PAS'
    Must match the identical function in compute_feats_clam.py.
    """
    return re.sub(r'[^\w\-]+', '_', stain).strip('_')


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
                    bag_name = f"{entry.name}/{slide_id}"  # matches discover_bags() format
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


# ---------------------------------------------------------------------------
# Per-patch analysis
# ---------------------------------------------------------------------------

def _patch_lab_stats(patch_np, bg_threshold):
    """Return (means, stds) arrays of shape (3,) for L, a, b, or None to skip.

    Skips if:
      - mean L* >= bg_threshold  →  background / glass
      - any channel std < 1e-6   →  completely flat, degenerate patch

    L* after lab_split is in [0, 100].  White background ≈ 100, tissue ≈ 40-75.
    patch_np: uint8 RGB numpy array (H, W, 3).
    """
    try:
        lab = rgb2lab(patch_np.astype("float32") / 255)
        channels = lab_split(lab)          # tuple: (L/2.55, a-128, b-128)
        if np.mean(channels[0]) >= bg_threshold:
            return None
        stats = np.array([get_mean_std(c) for c in channels])  # (3, 2): col0=mean, col1=std
        if np.any(stats[:, 1] < 1e-6):
            return None
        return stats[:, 0].astype(np.float32), stats[:, 1].astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def compute_reference(
    coord_files,
    wsi_dir,
    slide_ext,
    n_patches_per_slide,
    bg_threshold,
    rng,
    n_save=0,
):
    """Sample patches from the given slides, accumulate LAB stats, return grand mean.

    coord_files: list of (h5_path, biopsy_nr, slide_id) as produced by
                 _discover_coord_files (already filtered to the slides of interest).

    Returns (target_means, target_stds, qc_patches):
      target_means, target_stds — float32 numpy arrays of shape (3,).
      qc_patches — list of (slide_id, patch_np) for the first n_save valid
                   tissue patches encountered; empty when n_save=0.
    """
    all_means = []
    all_stds = []
    qc_patches = []
    n_slides_used = 0

    for h5_path, biopsy_nr, slide_id in tqdm(coord_files, desc="Slides"):
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
            with h5py.File(h5_path, "r") as f:
                coords = f["coords"][:]
                patch_size = int(f["coords"].attrs["patch_size"])
                patch_level = int(f["coords"].attrs["patch_level"])
        except Exception as exc:
            print(f"  [WARN] Cannot read {h5_path}: {exc}")
            wsi.close()
            continue

        n = len(coords)
        sample_idx = rng.choice(n, size=min(n_patches_per_slide, n), replace=False)

        slide_means, slide_stds = [], []
        for i in sample_idx:
            x, y = int(coords[i][0]), int(coords[i][1])
            try:
                patch = wsi.read_region(
                    (x, y), patch_level, (patch_size, patch_size)
                ).convert("RGB")
                patch_np = np.array(patch)
            except Exception:
                continue

            result = _patch_lab_stats(patch_np, bg_threshold)
            if result is None:
                continue
            slide_means.append(result[0])
            slide_stds.append(result[1])
            if len(qc_patches) < n_save:
                qc_patches.append((slide_id, patch_np.copy()))

        wsi.close()

        if slide_means:
            all_means.extend(slide_means)
            all_stds.extend(slide_stds)
            n_slides_used += 1

    if not all_means:
        raise RuntimeError(
            "No valid tissue patches found across all slides. "
            "Check --patches_dir, --wsi_dir, and --bg_threshold."
        )

    target_means = np.mean(all_means, axis=0).astype(np.float32)
    target_stds = np.mean(all_stds, axis=0).astype(np.float32)

    print(
        f"Reference built from {len(all_means)} tissue patches "
        f"across {n_slides_used}/{len(coord_files)} slides."
    )
    print(f"  target_means (L, a, b): {target_means}")
    print(f"  target_stds  (L, a, b): {target_stds}")

    return target_means, target_stds, qc_patches


# ---------------------------------------------------------------------------
# QC image saving
# ---------------------------------------------------------------------------

def _save_qc_images(qc_patches, target_means, target_stds, out_dir):
    """Save side-by-side before/after PNG images for each patch in qc_patches.

    Each image is named '{slide_id}_{i:03d}.png' and shows the original patch
    on the left, a 4-pixel white separator, and the normalised patch on the right.
    Patches where normalisation fails (degenerate statistics) are skipped with a warning.
    """
    from PIL import Image
    from torchstain.numpy.normalizers.reinhard import NumpyReinhardNormalizer

    normalizer = NumpyReinhardNormalizer(method="modified")
    normalizer.target_means = target_means
    normalizer.target_stds = target_stds

    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for i, (slide_id, patch_np) in enumerate(qc_patches):
        try:
            normalized = normalizer.normalize(patch_np)
        except Exception as exc:
            print(f"  [WARN] QC normalisation failed for patch {i} ({slide_id}): {exc}")
            continue

        h = patch_np.shape[0]
        separator = np.full((h, 4, 3), 255, dtype=np.uint8)
        side_by_side = np.concatenate([patch_np, separator, normalized], axis=1)

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
        description="Compute mean Modified Reinhard LAB reference from CLAM-tiled WSI patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patches_dir",
        required=True,
        help="Directory of CLAM patch-coordinate .h5 files (e.g. WSI/IgA/patches).",
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
        "--bg_threshold",
        type=float,
        default=85.0,
        help=(
            "Mean L* threshold (0–100) above which a patch is treated as background. "
            "Tissue is typically 40–75; white background is ~100."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible patch sampling.",
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
            "Each image shows a randomly selected tissue patch before (left) "
            "and after (right) Modified Reinhard normalisation. "
            "Omit to skip QC image saving."
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

    os.makedirs(args.output, exist_ok=True)
    n_save = args.n_save_patches if args.save_patches is not None else 0

    # Scan the filesystem once; a genuinely empty patches_dir is fatal, but
    # individual stains with no patched slides are skipped below (not fatal).
    all_coord_files = _discover_coord_files(args.patches_dir)
    if not all_coord_files:
        raise RuntimeError(f"No .h5 coord files found in {args.patches_dir!r}.")

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
            qc_dir = os.path.join(args.save_patches, stem) if args.save_patches else None
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
        target_means, target_stds, qc_patches = compute_reference(
            coord_files=coord_files,
            wsi_dir=args.wsi_dir,
            slide_ext=slide_ext,
            n_patches_per_slide=args.n_patches_per_slide,
            bg_threshold=args.bg_threshold,
            rng=rng,
            n_save=n_save,
        )
        if qc_dir is not None:
            _save_qc_images(qc_patches, target_means, target_stds, qc_dir)
        torch.save(
            {
                "target_means": torch.tensor(target_means, dtype=torch.float32),
                "target_stds": torch.tensor(target_stds, dtype=torch.float32),
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
