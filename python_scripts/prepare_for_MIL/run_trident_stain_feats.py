"""
run_trident_stain_feats.py — TRIDENT patch-feature extraction with per-stain normalisation.

Stain normalisation is per-stain, but TRIDENT's Processor reuses one encoder across all
slides.  We therefore GROUP slides by stain and run the ``feat`` stage once per stain group,
each time wrapping the TRIDENT encoder with that stain's reference (StainNormPatchEncoder).
Segmentation and patching are stain-independent, so run them ONCE with stock TRIDENT first:

    # 1. Segmentation + coordinates (stock TRIDENT, all slides, run once)
    python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py \\
        --task seg   --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --segmenter hest
    python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py \\
        --task coords --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident \\
        --mag 20 --patch_size 224 --overlap 0

    # 2. Per-stain feature extraction (this script)
    python python_scripts/prepare_for_MIL/run_trident_stain_feats.py \\
        --wsi_dir       data/raw_wsi/IgA \\
        --job_dir       WSI/IgA/trident \\
        --labels_csv    label_csvs/labels_unfiltered.csv \\
        --stain_refs_dir stain_refs/IgA \\
        --backbone      uni_v2 \\
        --mag 20 --patch_size 224 --overlap 0 --batch_size 256

Outputs (stock TRIDENT layout — features + coords in one .h5 per slide):
    {job_dir}/{mag}x_{ps}px_{ov}px_overlap/features_{enc_name}/{slide_id}.h5

Use reorganize_trident_feats.py afterwards to fold these into the biopsy-nested
{backbone}_feats/{biopsy_nr}/{slide_id}.h5 layout consumed by ProcessedMILDataset.

Notes:
    * Slides not present in --labels_csv, or whose stain has no reference .pt, are extracted
      WITHOUT stain normalisation (raw patches) and a warning is printed.
    * TRIDENT smart-resume: re-running skips slides whose feature .h5 already exists, so the
      per-stain runs over the same job_dir never conflict.
    * --enc_name_suffix keeps stain-normalised features in features_{backbone}{suffix}/ so a
      later raw-patch run doesn't overwrite them (default: no suffix → features_{backbone}/).
"""

import argparse
import os
import sys

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve TRIDENT package + this dir (for stain_norm_encoder)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_trident_dir = os.path.abspath(
    os.path.join(_script_dir, "..", "external_repositories", "TRIDENT-main")
)
if _trident_dir not in sys.path:
    sys.path.insert(0, _trident_dir)

from trident import Processor  # noqa: E402
from trident.patch_encoder_models.load import encoder_factory  # noqa: E402

from stain_norm_encoder import build_stain_encoder  # noqa: E402


def _sanitize_stain_name(stain: str) -> str:
    """Match fit_stain_reference.py / compute_feats_clam.py: 'IgG - IgM - IgA' -> 'IgG_-_IgM_-_IgA'."""
    import re

    return re.sub(r"[^\w\-]+", "_", stain).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Per-stain TRIDENT patch-feature extraction (seg+coords must be done first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wsi_dir", required=True, help="Directory of raw WSI files.")
    parser.add_argument(
        "--job_dir",
        required=True,
        help="TRIDENT job dir that already contains seg + coords output.",
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="CSV with 'file_name' and 'stain' columns (optional 'mpp').",
    )
    parser.add_argument(
        "--stain_refs_dir",
        required=True,
        help="Directory of .pt references from fit_stain_reference.py (one per stain).",
    )
    parser.add_argument(
        "--backbone",
        default="uni_v2",
        help="TRIDENT patch encoder name (e.g. uni_v2, virchow2, hoptimus1, hibou_l).",
    )
    parser.add_argument(
        "--enc_name_suffix",
        default="",
        help="Appended to the output folder name (features_{backbone}{suffix}/).",
    )
    parser.add_argument("--slide_ext", default=".svs", help="WSI file extension.")
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--search_nested",
        action="store_true",
        help="Recursively search --wsi_dir for slides (biopsy-nested raw-WSI layout).",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help=(
            "Continue to the next slide when one fails instead of aborting. Failures are "
            "logged by TRIDENT to {job_dir}/{coords_dir}/_logs_feats_{enc}.txt and "
            "{job_dir}/wsi_states/, with the offending filename."
        ),
    )
    parser.add_argument(
        "--patch_encoder_ckpt_path",
        default=None,
        help="Optional local encoder checkpoint (offline use).",
    )
    args = parser.parse_args()

    slide_ext = args.slide_ext if args.slide_ext.startswith(".") else f".{args.slide_ext}"
    device = (
        f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    )
    mag_str = f"{float(args.mag):g}"
    coords_dir = f"{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap"

    # --- Group slides by stain --------------------------------------------------
    df = pd.read_csv(args.labels_csv)
    if "file_name" not in df.columns or "stain" not in df.columns:
        parser.error("--labels_csv must contain 'file_name' and 'stain' columns.")
    has_mpp = "mpp" in df.columns

    # Build the base encoder once; it is re-wrapped per stain with a different reference.
    print(f"Loading TRIDENT encoder '{args.backbone}' …")
    base_encoder = encoder_factory(
        args.backbone, weights_path=args.patch_encoder_ckpt_path
    )
    base_encoder = base_encoder.eval().to(device)
    out_enc_name = f"{base_encoder.enc_name}{args.enc_name_suffix}"
    print(f"Output folder: features_{out_enc_name}/  |  device: {device}")

    os.makedirs(args.job_dir, exist_ok=True)
    tmp_dir = os.path.join(args.job_dir, "_stain_group_lists")
    os.makedirs(tmp_dir, exist_ok=True)

    stains = [s for s in df["stain"].dropna().unique()]
    print(f"Found {len(stains)} stain group(s): {stains}")

    for stain in stains:
        group = df[df["stain"] == stain]
        ref_path = os.path.join(args.stain_refs_dir, f"{_sanitize_stain_name(stain)}.pt")
        if not os.path.isfile(ref_path):
            print(
                f"\n[WARN] stain '{stain}': no reference at {ref_path} — "
                f"{len(group)} slide(s) will be extracted WITHOUT normalisation."
            )
            ref_path = None

        # Write a TRIDENT custom_list_of_wsis CSV (column 'wsi', optional 'mpp').
        rows = []
        for _, r in group.iterrows():
            row = {"wsi": f"{r['file_name']}{slide_ext}"}
            if has_mpp and pd.notna(r.get("mpp")):
                row["mpp"] = r["mpp"]
            rows.append(row)
        group_csv = os.path.join(tmp_dir, f"{_sanitize_stain_name(stain)}.csv")
        pd.DataFrame(rows).to_csv(group_csv, index=False)

        print(
            f"\n=== stain '{stain}'  |  {len(rows)} slide(s)  |  "
            f"norm: {'on' if ref_path else 'off'} ==="
        )

        # Wrap the encoder with this stain's reference (passthrough if ref_path is None).
        wrapped = build_stain_encoder(
            base_encoder,
            stain_ref_path=ref_path,
            device=torch.device(device),
            enc_name=out_enc_name,
        )

        processor = Processor(
            job_dir=args.job_dir,
            wsi_source=args.wsi_dir,
            custom_list_of_wsis=group_csv,
            max_workers=args.max_workers,
            search_nested=args.search_nested,
            skip_errors=args.skip_errors,
        )
        processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=wrapped,
            device=device,
            saveas="h5",
            batch_limit=args.batch_size,
        )

    print(
        f"\nDone.  Features → {os.path.join(args.job_dir, coords_dir, f'features_{out_enc_name}')}"
    )


if __name__ == "__main__":
    main()
