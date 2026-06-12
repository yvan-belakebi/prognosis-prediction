"""
run_clam_tiling.py — Tile WSI files using CLAM's segmentation + patching pipeline.

Output layout mirrors the biopsy-nested project structure:

    WSI/{dataset}/patches/{biopsy_nr}/{slide_id}.h5   ← patch coordinates (N, 2)
    WSI/{dataset}/masks/{biopsy_nr}/{slide_id}.jpg    ← tissue segmentation overlay
    WSI/{dataset}/stitches/{biopsy_nr}/{slide_id}.jpg ← patch grid stitch (optional)

Input WSIs are discovered from:
    --wsi_dir/{biopsy_nr}/{slide_id}.svs   (nested layout, one dir per biopsy)
    --wsi_dir/{slide_id}.svs               (flat layout, all slides in one dir)

The script auto-skips slides whose .h5 already exists in the output directory.

Usage (nested, PAS stain, 256 px patches at 20x):
    python python_scripts/prepare_for_MIL/run_clam_tiling.py \\
        --wsi_dir    data/raw_wsi/IgA \\
        --output_dir WSI/IgA \\
        --patch_size 256 --step_size 256 --patch_level 0 \\
        --slide_ext  .svs

Usage (flat, custom tissue thresholds):
    python python_scripts/prepare_for_MIL/run_clam_tiling.py \\
        --wsi_dir    data/raw_wsi/non_IgA \\
        --output_dir WSI/non_IgA \\
        --sthresh 15 --a_t 200
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Resolve CLAM package from sibling directory
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_clam_dir = os.path.abspath(os.path.join(_script_dir, "..", "CLAM-master"))
if _clam_dir not in sys.path:
    sys.path.insert(0, _clam_dir)

from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

# ---------------------------------------------------------------------------
# Slide discovery
# ---------------------------------------------------------------------------


def discover_slides(
    wsi_dir: str, slide_exts: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Return list of (wsi_path, biopsy_nr, slide_id) for all matching slides.

    Supports both flat (all slides in wsi_dir) and nested (wsi_dir/{biopsy_nr}/*.svs)
    layouts.  Nested dirs take priority: if any subdirectory contains slide files the
    flat entries in wsi_dir itself are ignored (they would be dataset-level metadata,
    not slides).
    """

    def _matches(name: str) -> bool:
        return any(name.lower().endswith(ext) for ext in slide_exts)

    results = []
    has_nested = False

    for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
        if entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file() and _matches(sub.name):
                    slide_id = os.path.splitext(sub.name)[0]
                    results.append((sub.path, entry.name, slide_id))
                    has_nested = True

    if not has_nested:
        # Flat layout: treat all slides as belonging to biopsy "" (no subdirectory)
        for entry in sorted(os.scandir(wsi_dir), key=lambda e: e.name):
            if entry.is_file() and _matches(entry.name):
                slide_id = os.path.splitext(entry.name)[0]
                results.append((entry.path, "", slide_id))

    return results


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------


def tile_one_slide(
    wsi_path: str,
    patch_save_dir: str,
    mask_save_dir: str,
    stitch_save_dir: str,
    patch_size: int,
    step_size: int,
    patch_level: int,
    seg_level: int,
    sthresh: int,
    mthresh: int,
    close: int,
    use_otsu: bool,
    a_t: int,
    a_h: int,
    max_n_holes: int,
    do_stitch: bool,
    auto_skip: bool,
) -> str:
    """Segment tissue and extract patch coordinates for one WSI.

    Returns a status string: 'done', 'skipped', or 'failed'.
    """
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    out_h5 = os.path.join(patch_save_dir, slide_id + ".h5")

    if auto_skip and os.path.isfile(out_h5):
        return "skipped"

    try:
        wsi_obj = WholeSlideImage(wsi_path)
    except Exception as exc:
        print(f"  [ERROR] Cannot open {wsi_path}: {exc}")
        return "failed"

    # Resolve seg_level from the slide if set to auto (-1)
    if seg_level < 0:
        try:
            best = wsi_obj.wsi.get_best_level_for_downsample(64)
        except Exception:
            best = 0
        seg_level = best

    # Safety check: refuse very large thumbnail levels
    try:
        w, h = wsi_obj.level_dim[seg_level]
        if w * h > 1e8:
            print(
                f"  [SKIP] Seg level {seg_level} dim {w}×{h} too large for "
                f"{slide_id} — try a higher seg_level value."
            )
            return "failed"
    except IndexError:
        seg_level = 0

    # Tissue segmentation
    try:
        wsi_obj.segmentTissue(
            seg_level=seg_level,
            sthresh=sthresh,
            mthresh=mthresh,
            close=close,
            use_otsu=use_otsu,
            filter_params={"a_t": a_t, "a_h": a_h, "max_n_holes": max_n_holes},
        )
    except Exception as exc:
        print(f"  [ERROR] Segmentation failed for {slide_id}: {exc}")
        return "failed"

    # Save tissue mask overlay
    try:
        vis_level = wsi_obj.wsi.get_best_level_for_downsample(64)
        mask_img = wsi_obj.visWSI(vis_level=vis_level, line_thickness=250)
        mask_img.save(os.path.join(mask_save_dir, slide_id + ".jpg"))
    except Exception as exc:
        print(f"  [WARN] Could not save mask for {slide_id}: {exc}")

    # Patch coordinate extraction
    try:
        wsi_obj.process_contours(
            save_path=patch_save_dir,
            patch_level=patch_level,
            patch_size=patch_size,
            step_size=step_size,
            use_padding=True,
            contour_fn="four_pt",
        )
    except Exception as exc:
        print(f"  [ERROR] Patching failed for {slide_id}: {exc}")
        return "failed"

    if not os.path.isfile(out_h5):
        print(f"  [WARN] No tissue found / no patches saved for {slide_id}.")
        return "failed"

    # Optional stitch visualisation
    if do_stitch:
        try:
            from wsi_core.wsi_utils import StitchCoords

            stitch = StitchCoords(
                file_path,
                wsi_obj,
                downscale=64,
                bg_color=(0, 0, 0),
                alpha=-1,
                draw_grid=False,
            )
            stitch.save(os.path.join(stitch_save_dir, slide_id + ".jpg"))
        except Exception as exc:
            print(f"  [WARN] Stitch failed for {slide_id}: {exc}")

    return "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CLAM-based WSI tiling with biopsy-nested output structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O -----------------------------------------------------------------
    parser.add_argument(
        "--wsi_dir",
        required=True,
        help="Root directory of raw WSI files. Nested: {wsi_dir}/{biopsy_nr}/*.svs; "
        "flat: {wsi_dir}/*.svs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root. Patches written to {output_dir}/patches/{biopsy_nr}/{slide}.h5.",
    )
    parser.add_argument(
        "--slide_ext",
        nargs="+",
        default=[".svs", ".ndpi"],
        help="WSI file extension(s) to include (case-insensitive). Multiple allowed.",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Save stitch visualisation (patch grid overlay) to {output_dir}/stitches/.",
    )
    parser.add_argument(
        "--no_auto_skip",
        action="store_true",
        help="Re-process slides whose .h5 already exists (default: skip).",
    )

    # --- Patching ------------------------------------------------------------
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch size in pixels at the extraction level.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=224,
        help="Step size between patches (equals patch_size for non-overlapping tiles).",
    )
    parser.add_argument(
        "--patch_level",
        type=int,
        default=0,
        help="OpenSlide pyramid level for patch extraction (0 = highest resolution).",
    )

    # --- Tissue segmentation -------------------------------------------------
    parser.add_argument(
        "--seg_level",
        type=int,
        default=-1,
        help="OpenSlide level used for tissue segmentation (-1 = auto-select ~64× downsample).",
    )
    parser.add_argument(
        "--sthresh",
        type=int,
        default=8,
        help="Saturation threshold for tissue/background segmentation.",
    )
    parser.add_argument(
        "--mthresh",
        type=int,
        default=7,
        help="Median blur kernel size for segmentation.",
    )
    parser.add_argument(
        "--close",
        type=int,
        default=4,
        help="Morphological closing iterations.",
    )
    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding instead of fixed sthresh.",
    )
    parser.add_argument(
        "--a_t",
        type=int,
        default=100,
        help="Minimum tissue contour area (in units of ref_patch_size² pixels).",
    )
    parser.add_argument(
        "--a_h",
        type=int,
        default=16,
        help="Minimum hole area to remove from tissue contours.",
    )
    parser.add_argument(
        "--max_n_holes",
        type=int,
        default=8,
        help="Maximum number of holes to consider per tissue contour.",
    )

    args = parser.parse_args()
    slide_exts = tuple(
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.slide_ext
    )

    # --- Discover slides -----------------------------------------------------
    print(
        f"Discovering slides in {args.wsi_dir} (extensions: {', '.join(slide_exts)}) …"
    )
    slides = discover_slides(args.wsi_dir, slide_exts)
    if not slides:
        print(f"No {'/'.join(slide_exts)} files found in {args.wsi_dir}.")
        return
    print(f"Found {len(slides)} slides.")

    # --- Process each slide --------------------------------------------------
    counts = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    for i, (wsi_path, biopsy_nr, slide_id) in enumerate(slides):
        biopsy_tag = f"[{biopsy_nr}] " if biopsy_nr else ""
        print(f"\n({i + 1}/{len(slides)}) {biopsy_tag}{slide_id}")

        # Build per-biopsy output dirs
        subdir = biopsy_nr if biopsy_nr else ""
        patch_save_dir = os.path.join(args.output_dir, "patches", subdir)
        mask_save_dir = os.path.join(args.output_dir, "masks", subdir)
        stitch_save_dir = os.path.join(args.output_dir, "stitches", subdir)
        for d in (patch_save_dir, mask_save_dir, stitch_save_dir):
            os.makedirs(d, exist_ok=True)

        status = tile_one_slide(
            wsi_path=wsi_path,
            patch_save_dir=patch_save_dir,
            mask_save_dir=mask_save_dir,
            stitch_save_dir=stitch_save_dir,
            patch_size=args.patch_size,
            step_size=args.step_size,
            patch_level=args.patch_level,
            seg_level=args.seg_level,
            sthresh=args.sthresh,
            mthresh=args.mthresh,
            close=args.close,
            use_otsu=args.use_otsu,
            a_t=args.a_t,
            a_h=args.a_h,
            max_n_holes=args.max_n_holes,
            do_stitch=args.stitch,
            auto_skip=not args.no_auto_skip,
        )
        counts[status] += 1
        print(f"  → {status}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s.  "
        f"Processed: {counts['done']}  Skipped: {counts['skipped']}  Failed: {counts['failed']}"
    )
    print(f"Patches → {os.path.join(args.output_dir, 'patches')}")


if __name__ == "__main__":
    main()
