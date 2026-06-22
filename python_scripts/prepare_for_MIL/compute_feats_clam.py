"""
compute_feats_clam.py — Extract patch features from CLAM-tiled WSIs.

Reads coordinate .h5 files produced by run_clam_tiling.py, opens the raw WSI
with OpenSlide, and extracts patches on-the-fly.  Features + coordinates are
stored together in one .h5 per slide (keys: 'features', 'coords'), which is the
format expected by ProcessedMILDataset with file_ext='.h5'.

Output layout:
    WSI/{dataset}/{backbone}_feats/{biopsy_nr}/{slide_id}.h5
        h5['features']  shape (N, D)   float32
        h5['coords']    shape (N, 2)   int32   pixel coords at patch_level 0

Input layouts supported:
    patches_dir/{biopsy_nr}/{slide_id}.h5   (nested, one dir per biopsy)
    patches_dir/{slide_id}.h5              (flat)

The corresponding raw WSI is looked up under:
    wsi_dir/{biopsy_nr}/{slide_id}.{slide_ext}   (nested)
    wsi_dir/{slide_id}.{slide_ext}              (flat)

Slides whose output .h5 already exists are skipped unless --no_auto_skip is set.

Usage:
    python python_scripts/prepare_for_MIL/compute_feats_clam.py \\
        --patches_dir WSI/IgA/patches \\
        --wsi_dir     data/raw_wsi/IgA \\
        --output_dir  WSI/IgA/UNI2-h_feats \\
        --backbone    UNI2-h \\
        --batch_size  256 --num_workers 8

Supported backbones:
    UNI2-h       (1536-D, requires offline weights via load_feature_extractors_offline)
    h-optimus-1  (1536-D)
    Virchow2     (2560-D)
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve sibling packages
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_clam_dir = os.path.abspath(os.path.join(_script_dir, "..", "CLAM-master"))
if _clam_dir not in sys.path:
    sys.path.insert(0, _clam_dir)

_staintools_dir = os.path.abspath(
    os.path.join(_script_dir, "..", "torch-staintools-main")
)
if _staintools_dir not in sys.path:
    sys.path.insert(0, _staintools_dir)

# CLAM's on-the-fly patch loader
import openslide  # noqa: E402 (requires openslide-python)
from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP  # noqa: E402
from utils.file_utils import save_hdf5  # noqa: E402
from torch_staintools.constants import CONFIG  # noqa: E402
from torch_staintools.normalizer import NormalizerBuilder  # noqa: E402

# torch.compile can speed up the per-patch stain-matrix estimation, but its backend
# needs a working compiler toolchain that is often absent or misconfigured (Triton +
# libcuda on Linux/GPU, MSVC 'cl.exe' on Windows) and then hard-crashes the run. It is
# therefore OFF by default and opt-in via --compile. When enabled, the loop below pads
# the short final batch up to batch_size so the normalizer sees one shape and compiles
# once for the whole run instead of recompiling per slide.
CONFIG.ENABLE_COMPILE = False

# Concentration solver — see fit_stain_reference.py. 'qr' is robust for batched tiles.
_CONCENTRATION_SOLVER = "qr"
_LUMINOSITY_THRESHOLD = 0.8
# Lowered solver work (factory defaults are 60 / 50). Vahadane dictionary learning
# converges early, so fewer iterations trade a little accuracy for substantially faster
# per-patch stain-matrix estimation.
_SPARSE_DICT_STEPS = 30
_MAXITER = 30

# ---------------------------------------------------------------------------
# Stain normalisation (Vahadane/Macenko, torch-staintools — applied on the GPU batch)
# ---------------------------------------------------------------------------


def load_stain_normalizer(ref_path: str, device: torch.device):
    """Build a stain normalizer from a .pt reference produced by fit_stain_reference.py.

    The algorithm is auto-matched to the 'method' recorded in the reference (defaulting
    to 'vahadane' for older references that predate the key), so the per-patch source
    stain matrix is estimated the same way as the target was. The fitted target buffers
    ('stain_matrix_target', 'maxC_target') are re-injected so it can transform without
    re-fitting. Returns (nn.Module, method); the module maps an RGB [0, 1] BxCxHxW batch
    to a normalised one and returns the input unchanged on a degenerate/empty-mask patch.
    """
    ref = torch.load(ref_path, weights_only=True)
    method = ref.get("method", "vahadane")
    kwargs = dict(
        concentration_solver=_CONCENTRATION_SOLVER,
        maxiter=_MAXITER,
        luminosity_threshold=_LUMINOSITY_THRESHOLD,
        device=device,
    )
    if method == "vahadane":
        kwargs["sparse_dict_steps"] = _SPARSE_DICT_STEPS  # Vahadane-only knob
    norm = NormalizerBuilder.build(method, **kwargs).to(device)
    norm.register_buffer("stain_matrix_target", ref["stain_matrix_target"].to(device))
    norm.register_buffer("maxC_target", ref["maxC_target"].to(device))
    return norm, method


def split_backbone_transform(base_transform):
    """Split a backbone torchvision Compose at the trailing Normalize.

    Returns (pre_transform, backbone_normalize) where pre_transform turns a PIL patch
    into an RGB [0, 1] CxHxW tensor (Resize/CenterCrop/ToTensor) and backbone_normalize
    is the mean/std Normalize. This lets stain normalisation run on RGB [0, 1] (on the
    GPU batch) before the backbone's channel normalisation.
    """
    from torchvision import transforms as T

    steps = list(base_transform.transforms)
    if not steps or not isinstance(steps[-1], T.Normalize):
        raise ValueError(
            "Expected the backbone transform to end with a torchvision Normalize; "
            f"got {type(steps[-1]).__name__ if steps else 'empty Compose'}."
        )
    return T.Compose(steps[:-1]), steps[-1]


def _sanitize_stain_name(stain: str) -> str:
    """Convert a stain name to a safe filename stem.

    e.g. 'IgG - IgM - IgA' → 'IgG_-_IgM_-_IgA',  'PAS' → 'PAS'
    Must match the identical function in fit_stain_reference.py.
    """
    import re

    return re.sub(r"[^\w\-]+", "_", stain).strip("_")


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------


def load_backbone(backbone: str):
    """Return (model, transform) for the requested backbone."""
    if backbone == "UNI2-h":
        from load_feature_extractors_offline import load_uni2h_feature_extractor

        return load_uni2h_feature_extractor()
    if backbone == "h-optimus-1":
        from load_feature_extractors_offline import load_hoptimus1_feature_extractor

        return load_hoptimus1_feature_extractor()
    if backbone == "Virchow2":
        from load_feature_extractors_offline import load_virchow2_feature_extractor

        return load_virchow2_feature_extractor()
    raise ValueError(
        f"Unknown backbone '{backbone}'. " "Choose from: UNI2-h, h-optimus-1, Virchow2."
    )


# ---------------------------------------------------------------------------
# Slide discovery
# ---------------------------------------------------------------------------


def discover_coord_files(patches_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (h5_path, biopsy_nr, slide_id).

    Supports flat (patches_dir/*.h5) and nested (patches_dir/{biopsy}/*.h5) layouts.
    Nested dirs take priority (same rule as run_clam_tiling.py).
    """
    results = []
    has_nested = False

    for entry in sorted(os.scandir(patches_dir), key=lambda e: e.name):
        if entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file() and sub.name.endswith(".h5"):
                    slide_id = os.path.splitext(sub.name)[0]
                    results.append((sub.path, entry.name, slide_id))
                    has_nested = True

    if not has_nested:
        for entry in sorted(os.scandir(patches_dir), key=lambda e: e.name):
            if entry.is_file() and entry.name.endswith(".h5"):
                slide_id = os.path.splitext(entry.name)[0]
                results.append((entry.path, "", slide_id))

    return results


def find_wsi(wsi_dir: str, biopsy_nr: str, slide_id: str, slide_ext: str) -> str | None:
    """Locate the raw WSI file for a given slide."""
    candidates = []
    # Nested first
    if biopsy_nr:
        candidates.append(os.path.join(wsi_dir, biopsy_nr, slide_id + slide_ext))
    # Flat fallback
    candidates.append(os.path.join(wsi_dir, slide_id + slide_ext))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Per-slide feature extraction
# ---------------------------------------------------------------------------


def extract_slide_features(
    h5_path: str,
    wsi_path: str,
    model: torch.nn.Module,
    pre_transform,
    backbone_normalize,
    stain_normalizer,
    output_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str = "",
) -> int:
    """Extract features for one slide and write to output_path.

    Features are streamed to a sibling ``<output_path>.partial`` file and only
    renamed onto the final ``output_path`` once the whole slide has been written.
    The rename is atomic (``os.replace``), so an abrupt interruption can leave a
    stray ``.partial`` file but never a truncated ``output_path`` — keeping the
    auto-skip logic in ``main`` trustworthy.

    Returns the number of patches processed.
    """
    try:
        wsi = openslide.open_slide(wsi_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open WSI {wsi_path}: {exc}") from exc

    tmp_path = output_path + ".partial"
    committed = False
    try:
        try:
            dataset = Whole_Slide_Bag_FP(
                file_path=h5_path,
                wsi=wsi,
                img_transforms=pre_transform,
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot load coord file {h5_path}: {exc}") from exc

        # OpenSlide objects contain ctypes pointers and cannot be pickled for
        # multiprocessing workers on Windows — force single-process loading.
        if sys.platform == "win32":
            num_workers = 0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Discard any leftover partial from a previous interrupted run.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        mode = "w"
        total_patches = 0
        with torch.inference_mode():
            for batch in tqdm(
                loader,
                desc=desc or "  patches",
                unit="batch",
                leave=False,
            ):
                imgs = batch["img"].to(device, non_blocking=True)  # RGB [0,1] BxCxHxW
                coords = batch["coord"].numpy().astype(np.int32)  # (B, 2)

                # Stain normalisation on the GPU batch (RGB [0,1] -> RGB [0,1]),
                # then the backbone's channel normalisation, then the forward pass.
                if stain_normalizer is not None:
                    b = imgs.shape[0]
                    if CONFIG.ENABLE_COMPILE and b < batch_size:
                        # Pad the short final batch up to batch_size so the compiled
                        # normalizer only ever sees one input shape (avoids
                        # recompilation); padding rows are dropped before the backbone.
                        pad = imgs[-1:].repeat(batch_size - b, 1, 1, 1)
                        imgs = stain_normalizer(torch.cat([imgs, pad], dim=0))[:b]
                    else:
                        imgs = stain_normalizer(imgs)
                imgs = backbone_normalize(imgs)
                feats = model(imgs)
                # Transformer models may output (B, seq, D) — flatten to (B*seq, D)
                if feats.ndim == 3:
                    feats = feats.reshape(-1, feats.shape[-1])
                feats_np = feats.cpu().numpy().astype(np.float32)

                save_hdf5(tmp_path, {"features": feats_np, "coords": coords}, mode=mode)
                mode = "a"
                total_patches += len(feats_np)

        # Atomically commit the fully-written file onto the final path.
        os.replace(tmp_path, output_path)
        committed = True
        return total_patches
    finally:
        wsi.close()
        # Drop the partial file unless it was successfully committed.
        if not committed and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract patch features from CLAM-tiled WSIs and save as .h5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O -----------------------------------------------------------------
    parser.add_argument(
        "--patches_dir",
        required=True,
        help="Directory of CLAM patch-coordinate .h5 files " "(e.g. WSI/IgA/patches).",
    )
    parser.add_argument(
        "--wsi_dir",
        required=True,
        help="Root directory of raw WSI files (e.g. data/raw_wsi/IgA).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for feature .h5 files " "(e.g. WSI/IgA/UNI2-h_feats).",
    )
    parser.add_argument(
        "--slide_ext",
        default=".svs",
        help="WSI file extension.",
    )
    parser.add_argument(
        "--no_auto_skip",
        action="store_true",
        help="Re-extract slides whose output .h5 already exists.",
    )

    # --- Model ---------------------------------------------------------------
    parser.add_argument(
        "--backbone",
        default="UNI2-h",
        choices=["UNI2-h", "h-optimus-1", "Virchow2"],
        help="Feature extractor backbone.",
    )

    # --- Stain normalisation -------------------------------------------------
    parser.add_argument(
        "--labels_csv",
        default=None,
        help=(
            "CSV with 'file_name' and 'stain' columns. "
            "Required together with --stain_refs_dir to enable per-slide stain normalisation."
        ),
    )
    parser.add_argument(
        "--stain_refs_dir",
        default=None,
        help=(
            "Directory of .pt reference files produced by fit_stain_reference.py, "
            "one per stain (e.g. PAS.pt, IgG_-_IgM_-_IgA.pt). "
            "Required together with --labels_csv."
        ),
    )

    # --- Compute -------------------------------------------------------------
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Patch batch size for GPU inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile for the stain normalizer (off by default). Speeds up "
            "stain normalisation but requires a working compile toolchain: Triton + "
            "libcuda on Linux/GPU, MSVC 'cl.exe' on Windows. If it errors with a "
            "Triton/gcc or 'compiler not found' message, drop this flag."
        ),
    )

    args = parser.parse_args()

    # torch.compile is off at import; honour --compile before any normalizer runs.
    if args.compile:
        CONFIG.ENABLE_COMPILE = True

    slide_ext = args.slide_ext
    if not slide_ext.startswith("."):
        slide_ext = f".{slide_ext}"

    device = torch.device(
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}  |  torch.compile: {CONFIG.ENABLE_COMPILE}")

    if (args.labels_csv is None) != (args.stain_refs_dir is None):
        parser.error("--labels_csv and --stain_refs_dir must be provided together.")

    # --- Load model ----------------------------------------------------------
    print(f"Loading backbone: {args.backbone} …")
    model, base_transform = load_backbone(args.backbone)
    model = model.eval().to(device)
    print(f"Backbone ready.")

    # --- Split the backbone transform at the trailing Normalize --------------
    # pre_transform (PIL → RGB [0,1] tensor) runs in the DataLoader; backbone_normalize
    # (mean/std) runs on the GPU batch after stain normalisation.
    pre_transform, backbone_normalize = split_backbone_transform(base_transform)

    # --- Build per-stain normalizer cache (method auto-matched per reference) -
    stain_lookup = {}  # bag_name (biopsy_nr/slide_id or slide_id) → stain label
    normalizer_cache = {}  # stain label → stain-normalizer nn.Module (on device)
    method_cache = {}  # stain label → method name ('vahadane' / 'macenko')

    if args.labels_csv is not None:
        import pandas as pd

        df_labels = pd.read_csv(args.labels_csv)
        stain_lookup = dict(
            zip(df_labels["file_name"].astype(str), df_labels["stain"].astype(str))
        )
        n_loaded = 0
        methods_loaded = set()
        for stain in df_labels["stain"].dropna().unique():
            ref_path = os.path.join(
                args.stain_refs_dir, f"{_sanitize_stain_name(stain)}.pt"
            )
            if os.path.isfile(ref_path):
                norm, method = load_stain_normalizer(ref_path, device)
                normalizer_cache[stain] = norm
                method_cache[stain] = method
                methods_loaded.add(method)
                n_loaded += 1
            else:
                print(
                    f"  [WARN] No reference for stain '{stain}' ({ref_path}) — "
                    "normalisation skipped for affected slides."
                )
        methods_str = ", ".join(sorted(methods_loaded)) if methods_loaded else "none"
        print(
            f"Stain normalisation: ENABLED ({methods_str}) — "
            f"{n_loaded}/{df_labels['stain'].nunique()} stains loaded."
        )
    else:
        print(
            "Stain normalisation: DISABLED — features computed on raw patches. "
            "Pass --labels_csv and --stain_refs_dir to enable it."
        )

    # --- Discover coord files ------------------------------------------------
    print(f"Scanning {args.patches_dir} …")
    coord_files = discover_coord_files(args.patches_dir)
    if not coord_files:
        print("No .h5 coord files found.")
        return
    print(f"Found {len(coord_files)} slides.")

    # --- Sweep stale partials from previously interrupted runs ----------------
    n_swept = 0
    for root, _dirs, files in os.walk(args.output_dir):
        for name in files:
            if name.endswith(".h5.partial"):
                try:
                    os.remove(os.path.join(root, name))
                    n_swept += 1
                except OSError:
                    pass
    if n_swept:
        print(f"Removed {n_swept} stale .partial file(s) from interrupted runs.")

    # --- Process each slide --------------------------------------------------
    counts = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    slide_bar = tqdm(coord_files, desc="Slides", unit="slide")
    for i, (h5_path, biopsy_nr, slide_id) in enumerate(slide_bar):
        biopsy_tag = f"[{biopsy_nr}] " if biopsy_nr else ""
        slide_bar.set_postfix_str(f"{biopsy_tag}{slide_id}")
        tqdm.write(f"\n({i + 1}/{len(coord_files)}) {biopsy_tag}{slide_id}")

        subdir = biopsy_nr if biopsy_nr else ""
        output_path = os.path.join(args.output_dir, subdir, slide_id + ".h5")

        if not args.no_auto_skip and os.path.isfile(output_path):
            tqdm.write("  → skipped (output exists)")
            counts["skipped"] += 1
            continue

        # Per-slide stain normalisation
        lookup_key = f"{biopsy_nr}/{slide_id}" if biopsy_nr else slide_id
        if stain_lookup:
            stain = stain_lookup.get(lookup_key)
            if stain is None:
                tqdm.write(
                    f"  [WARN] {lookup_key!r} not in labels CSV — normalisation skipped."
                )
                stain_normalizer = None
            else:
                stain_normalizer = normalizer_cache.get(stain)
                if stain_normalizer is None:
                    tqdm.write(f"  stain: {stain}  |  norm: none (no reference)")
                else:
                    tqdm.write(f"  stain: {stain}  |  norm: {method_cache[stain]}")
        else:
            stain_normalizer = None

        wsi_path = find_wsi(args.wsi_dir, biopsy_nr, slide_id, slide_ext)
        if wsi_path is None:
            tqdm.write(
                f"  [WARN] Raw WSI not found for {slide_id} "
                f"(looked in {args.wsi_dir}/{biopsy_nr}/ and {args.wsi_dir}/)"
            )
            counts["failed"] += 1
            continue

        try:
            n_patches = extract_slide_features(
                h5_path=h5_path,
                wsi_path=wsi_path,
                model=model,
                pre_transform=pre_transform,
                backbone_normalize=backbone_normalize,
                stain_normalizer=stain_normalizer,
                output_path=output_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                desc=f"  {slide_id}",
            )
            tqdm.write(f"  → done  ({n_patches} patches)  → {output_path}")
            counts["done"] += 1
        except Exception as exc:
            # extract_slide_features commits atomically, so output_path is never
            # left half-written; any partial is cleaned up there on the way out.
            tqdm.write(f"  [ERROR] {exc}")
            counts["failed"] += 1
        except BaseException:
            # Ctrl-C / SIGTERM-as-exception: the .partial file is removed by
            # extract_slide_features' finally block; stop the run cleanly.
            slide_bar.close()
            tqdm.write("\n[INTERRUPTED] stopping — committed slides are intact.")
            raise

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s.  "
        f"Extracted: {counts['done']}  Skipped: {counts['skipped']}  Failed: {counts['failed']}"
    )
    print(f"Features → {args.output_dir}")


if __name__ == "__main__":
    main()
