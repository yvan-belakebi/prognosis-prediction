"""
extract_rotated_feats.py — Extract patch features with every patch rotated by 90°, on a random
subset of k slides, to test a foundation model's rotational invariance.

The same TRIDENT coords are reused, so rotated and unrotated features are row-aligned: row i
of both files is the same patch (same coords), once as-is and once rotated.

Seg + coords must already exist in --job_dir. Slides are found by walking --wsi_dir
recursively (biopsy-nested layout, .svs and .ndpi), e.g. on the registry:

    python python_scripts/prepare_for_MIL/extract_rotated_feats.py \\
        --wsi_dir /forskning/hbe/2023-517496/RegistryWSIs \\
        --job_dir /forskning/hbe/2023-517496/RegistryTilesAndFeatures \\
        --k 20 --seed 0 --backbone uni_v2 \\
        --patch_encoder_ckpt_path models/uni2h/pytorch_model.bin

Outputs:
    {sample_csv}  (default {job_dir}/rot_sample_k{k}_seed{seed}.csv) — sampled slides
                  (columns: wsi = path relative to --wsi_dir, slide_id). If it already
                  exists it is reused as-is, so re-runs resume on the same slides.
    {job_dir}/{coords_dir}/features_{enc}_rot{90*n_rot}/{slide_id}.h5 — rotated features
    {job_dir}/{coords_dir}/features_{enc}_rot0/{slide_id}.h5          — unrotated features

The unrotated features get their own _rot0 folder (not features_{enc}/) so they are never
mixed up with an existing, possibly stain-normalised, extraction: both sets go through the
exact same pipeline, raw patches without stain normalisation.

Slides are read straight from --wsi_dir (no local staging), which is fine for a small k.
"""

import argparse
import os
import random
import sys

import pandas as pd
import torch

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

from stain_feats_staged import index_wsis  # noqa: E402
from trident_io import coords_dir_name, discover_coords, patches_dir  # noqa: E402


class RotatedPatchEncoder(torch.nn.Module):
    """Rotate every patch of the batch by n_rot * 90° (counter-clockwise) before the backbone.

    The batch arrives already transformed (resize/crop + per-channel Normalize), and those
    commute with a 90° rotation of a square patch, so rotating here equals rotating the
    raw patch. n_rot=0 is the unrotated baseline.
    """

    def __init__(self, base_encoder, n_rot: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.n_rot = n_rot
        # Attributes TRIDENT's Processor / extract loop read.
        self.eval_transforms = base_encoder.eval_transforms
        self.precision = getattr(base_encoder, "precision", torch.float32)
        self.enc_name = f"{base_encoder.enc_name}_rot{90 * n_rot}"
        self.embedding_dim = getattr(base_encoder, "embedding_dim", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_encoder(torch.rot90(x, self.n_rot, dims=(2, 3)))


def sample_slides(args, coords_dir):
    """Pick k slides that have coords; write them to --sample_csv (or reuse it if present)."""
    if os.path.isfile(args.sample_csv):
        df = pd.read_csv(args.sample_csv)
        print(f"Reusing existing sample: {args.sample_csv} ({len(df)} slide(s)).")
        return df

    wsi_index = index_wsis(args.wsi_dir)  # {slide_id: path relative to wsi_dir}
    with_coords = discover_coords(patches_dir(args.job_dir, coords_dir))
    candidates = sorted((rel, sid) for sid, rel in wsi_index.items() if sid in with_coords)
    print(f"{len(wsi_index)} slide(s) under {args.wsi_dir}, {len(candidates)} with coords.")
    if args.k > len(candidates):
        sys.exit(f"--k {args.k} > {len(candidates)} slides with coords.")

    picked = random.Random(args.seed).sample(candidates, args.k)
    df = pd.DataFrame(picked, columns=["wsi", "slide_id"])
    os.makedirs(os.path.dirname(os.path.abspath(args.sample_csv)), exist_ok=True)
    df.to_csv(args.sample_csv, index=False)
    print(f"Sampled {args.k} slide(s) (seed {args.seed}) -> {args.sample_csv}")
    return df


def main():
    p = argparse.ArgumentParser(
        description="Rotated-patch feature extraction on a random subset of slides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wsi_dir", required=True, help="Root holding the WSIs (searched recursively).")
    p.add_argument("--job_dir", required=True, help="TRIDENT job dir with seg + coords.")
    p.add_argument("--k", type=int, required=True, help="Number of slides to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_csv", default=None,
                   help="Where to store the sampled slide names "
                        "(default {job_dir}/rot_sample_k{k}_seed{seed}.csv).")
    p.add_argument("--n_rot", type=int, default=1, help="Number of 90° CCW rotations.")
    p.add_argument("--skip_unrotated", action="store_true",
                   help="Only extract rotated features (no _rot0 pass).")
    p.add_argument("--backbone", default="uni_v2")
    p.add_argument("--patch_encoder_ckpt_path", default=None)
    p.add_argument("--mag", type=float, default=20.0)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--max_workers", type=int, default=None)
    p.add_argument("--skip_errors", action="store_true")
    args = p.parse_args()

    coords_dir = coords_dir_name(args.mag, args.patch_size, args.overlap)
    args.sample_csv = args.sample_csv or os.path.join(
        args.job_dir, f"rot_sample_k{args.k}_seed{args.seed}.csv"
    )
    device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"

    sample_slides(args, coords_dir)

    base = encoder_factory(args.backbone, weights_path=args.patch_encoder_ckpt_path)
    base = base.eval().to(device)
    rotations = [args.n_rot] if args.skip_unrotated else [args.n_rot, 0]

    for n_rot in rotations:
        enc = RotatedPatchEncoder(base, n_rot)
        print(f"\n=== features_{enc.enc_name} ===")
        Processor(
            job_dir=args.job_dir,
            wsi_source=args.wsi_dir,
            custom_list_of_wsis=args.sample_csv,  # 'wsi' = nested relative paths
            max_workers=args.max_workers,
            skip_errors=args.skip_errors,
        ).run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=enc,
            device=device,
            saveas="h5",
            batch_limit=args.batch_size,
        )


if __name__ == "__main__":
    main()
