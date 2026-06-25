"""
benchmark_stain_norm.py — Speed benchmark: Macenko vs Vahadane stain normalisation.

Runs the exact run_trident_stain_feats.py extraction path on a SMALL SUBSET of slides,
once with the Macenko references (--macenko_refs_dir, default stain_refs_macenko) and once
with the Vahadane references (--vahadane_refs_dir, default stain_refs_vahadane), and reports
wall-clock time, throughput (patches/s, slides/s) and the Vahadane/Macenko slowdown factor.

The two methods share the same backbone encoder and the same patches, so the only thing that
differs is the per-patch source stain-matrix estimation (Macenko: SVD + percentile; Vahadane:
sparse-NMF dictionary learning). To isolate that cost the benchmark:

    * loads the TRIDENT backbone ONCE and re-wraps it per method (load_stain_normalizer),
      so encoder-load time is excluded from the measured window;
    * writes each method's features to a throwaway folder features_{enc}_bench_{method}/ and
      deletes the subset's .h5 before every timed repeat, so TRIDENT smart-resume never skips
      a slide mid-benchmark (and the real features_{enc}/ output is left untouched);
    * times only Processor.run_patch_feature_extraction_job (seg + coords are reused).

Prerequisites (same as run_trident_stain_feats.py): seg + coords already done for the slides
under --job_dir, and one .pt reference per stain in BOTH refs dirs, e.g.

    python python_scripts/prepare_for_MIL/fit_stain_reference.py ... --method macenko  \\
        --output stain_refs_macenko/IgA
    python python_scripts/prepare_for_MIL/fit_stain_reference.py ... --method vahadane \\
        --output stain_refs_vahadane/IgA

Usage:
    python python_scripts/prepare_for_MIL/benchmark_stain_norm.py \\
        --wsi_dir         data/raw_wsi/IgA \\
        --job_dir         WSI/IgA/trident \\
        --labels_csv      label_csvs/labels_unfiltered.csv \\
        --macenko_refs_dir  stain_refs_macenko/IgA \\
        --vahadane_refs_dir stain_refs_vahadane/IgA \\
        --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 \\
        --n_slides 4 --batch_size 256 --search_nested
"""

import argparse
import os
import shutil
import sys
import time

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve TRIDENT package + this dir (mirror run_trident_stain_feats.py)
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

from run_trident_stain_feats import _sanitize_stain_name  # noqa: E402
from stain_norm_encoder import build_stain_encoder  # noqa: E402
from trident_io import (  # noqa: E402
    coords_dir_name,
    count_patches,
    discover_coords,
    feature_path,
    features_dir,
    patches_dir,
)

_METHODS = ("macenko", "vahadane")


def _select_subset(df, n_slides, seed):
    """Pick up to n_slides rows, spread across stains so each group is represented.

    Round-robins one slide per stain at a time (deterministic given --seed) so a small
    --n_slides still exercises every stain's reference rather than a single group.
    """
    rng = __import__("numpy").random.default_rng(seed)
    by_stain = {}
    for stain, grp in df.groupby("stain"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        by_stain[stain] = list(idx)

    chosen = []
    while len(chosen) < n_slides and any(by_stain.values()):
        for stain in list(by_stain):
            if not by_stain[stain]:
                continue
            chosen.append(by_stain[stain].pop(0))
            if len(chosen) >= n_slides:
                break
    return df.loc[chosen]


def _count_patches(job_dir, coords_dir, file_names):
    """Sum patch counts for the subset, pairing slides to coord files by on-disk scan.

    Uses discover_coords (which strips TRIDENT's '_patches' suffix from the files that
    actually exist) instead of building '{name}_patches.h5' from the labels CSV, so a
    file_name that does not match any coord-file stem is reported explicitly rather than
    silently counted as missing.

    Returns (total_patches, found_names, missing_names).
    """
    available = discover_coords(patches_dir(job_dir, coords_dir))
    total = 0
    found, missing = [], []
    for name in file_names:
        path = available.get(name)
        if path is None:
            missing.append(name)
            continue
        total += count_patches(path)
        found.append(name)
    return total, found, missing


def _clear_features(job_dir, coords_dir, enc_name, file_names):
    """Delete the benchmark feature .h5 for the subset so smart-resume recomputes them."""
    for name in file_names:
        fp = feature_path(job_dir, coords_dir, enc_name, name)
        if os.path.isfile(fp):
            os.remove(fp)


def _time_method(
    method,
    refs_dir,
    base_encoder,
    df_subset,
    args,
    coords_dir,
    device,
    slide_ext,
    tmp_dir,
):
    """Run + time the extraction for one method; returns elapsed seconds for one pass."""
    enc_name = f"{base_encoder.enc_name}_bench_{method}"
    file_names = [str(r["file_name"]) for _, r in df_subset.iterrows()]
    has_mpp = "mpp" in df_subset.columns

    # Force recompute: drop any features from a previous repeat/run.
    _clear_features(args.job_dir, coords_dir, enc_name, file_names)

    elapsed = 0.0
    for stain, group in df_subset.groupby("stain"):
        ref_path = os.path.join(refs_dir, f"{_sanitize_stain_name(stain)}.pt")
        if not os.path.isfile(ref_path):
            print(
                f"  [WARN] {method}: no reference for stain '{stain}' at {ref_path} — "
                f"this group runs WITHOUT normalisation (timing not comparable)."
            )
            ref_path = None

        rows = []
        for _, r in group.iterrows():
            row = {"wsi": f"{r['file_name']}{slide_ext}"}
            if has_mpp and pd.notna(r.get("mpp")):
                row["mpp"] = r["mpp"]
            rows.append(row)
        group_csv = os.path.join(tmp_dir, f"{method}_{_sanitize_stain_name(stain)}.csv")
        pd.DataFrame(rows).to_csv(group_csv, index=False)

        wrapped = build_stain_encoder(
            base_encoder,
            stain_ref_path=ref_path,
            device=torch.device(device),
            enc_name=enc_name,
        )
        processor = Processor(
            job_dir=args.job_dir,
            wsi_source=args.wsi_dir,
            custom_list_of_wsis=group_csv,
            max_workers=args.max_workers,
            search_nested=args.search_nested,
            skip_errors=args.skip_errors,
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=wrapped,
            device=device,
            saveas="h5",
            batch_limit=args.batch_size,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0

    return elapsed, enc_name


def main():
    parser = argparse.ArgumentParser(
        description="Speed benchmark of Macenko vs Vahadane stain normalisation on a subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wsi_dir", required=True, help="Directory of raw WSI files.")
    parser.add_argument(
        "--job_dir", required=True, help="TRIDENT job dir with seg + coords already done."
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="CSV with 'file_name' and 'stain' columns (optional 'mpp').",
    )
    parser.add_argument(
        "--macenko_refs_dir",
        default="stain_refs_macenko",
        help="Directory of Macenko .pt references (one per stain).",
    )
    parser.add_argument(
        "--vahadane_refs_dir",
        default="stain_refs_vahadane",
        help="Directory of Vahadane .pt references (one per stain).",
    )
    parser.add_argument("--backbone", default="uni_v2", help="TRIDENT patch encoder name.")
    parser.add_argument("--slide_ext", default=".svs", help="WSI file extension.")
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--n_slides",
        type=int,
        default=4,
        help="Number of slides in the benchmark subset (spread across stains).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed passes per method; the fastest pass is reported (less noisy).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducible subset selection."
    )
    parser.add_argument(
        "--keep_outputs",
        action="store_true",
        help="Keep the throwaway features_{enc}_bench_{method}/ dirs (default: delete).",
    )
    parser.add_argument(
        "--search_nested",
        action="store_true",
        help="Recursively search --wsi_dir for slides (biopsy-nested raw-WSI layout).",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Continue to the next slide when one fails instead of aborting.",
    )
    parser.add_argument(
        "--patch_encoder_ckpt_path",
        default=None,
        help="Optional local encoder checkpoint (offline use).",
    )
    args = parser.parse_args()

    slide_ext = args.slide_ext if args.slide_ext.startswith(".") else f".{args.slide_ext}"
    device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu"
    coords_dir = coords_dir_name(args.mag, args.patch_size, args.overlap)

    df = pd.read_csv(args.labels_csv)
    if "file_name" not in df.columns or "stain" not in df.columns:
        parser.error("--labels_csv must contain 'file_name' and 'stain' columns.")
    df = df.dropna(subset=["stain"])

    df_subset = _select_subset(df, args.n_slides, args.seed)
    file_names = [str(r["file_name"]) for _, r in df_subset.iterrows()]
    n_patches, found, missing = _count_patches(args.job_dir, coords_dir, file_names)
    n_with_coords = len(found)

    print("=" * 70)
    print("STAIN NORMALISATION SPEED BENCHMARK — Macenko vs Vahadane")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Backbone        : {args.backbone}")
    print(f"Coords dir      : {coords_dir}")
    print(f"Subset          : {len(df_subset)} slide(s) across "
          f"{df_subset['stain'].nunique()} stain(s)")
    print(f"Coord files     : {n_with_coords}/{len(df_subset)} found  "
          f"({n_patches} patches total)")
    print(f"Batch size      : {args.batch_size}  |  repeats: {args.repeats}")
    for _, r in df_subset.iterrows():
        flag = "" if str(r["file_name"]) in found else "   [no coord file]"
        print(f"   - {r['file_name']}  [{r['stain']}]{flag}")

    if missing:
        pdir = patches_dir(args.job_dir, coords_dir)
        present = sorted(discover_coords(pdir))
        print(
            f"\n[WARN] {len(missing)} of {len(df_subset)} subset slide(s) have no coord "
            f"file in {pdir}."
        )
        print(
            f"       {len(present)} coord file(s) exist there"
            + (f"; e.g. {present[:3]}" if present else " (directory empty or missing)")
            + "."
        )
        print(
            "       Likely causes: seg+coords were run at a different "
            "--mag/--patch_size/--overlap (the coords_dir must match exactly), or the "
            "labels file_name does not match the WSI basename."
        )
    if n_with_coords == 0:
        print(
            "\n[ERROR] No coord files matched the subset — cannot benchmark. "
            "Run seg + coords for these slides at this coords_dir first."
        )
        return 1

    print(f"\nLoading TRIDENT encoder '{args.backbone}' once …")
    base_encoder = encoder_factory(
        args.backbone, weights_path=args.patch_encoder_ckpt_path
    )
    base_encoder = base_encoder.eval().to(device)

    tmp_dir = os.path.join(args.job_dir, "_bench_stain_lists")
    os.makedirs(tmp_dir, exist_ok=True)

    refs_dirs = {
        "macenko": args.macenko_refs_dir,
        "vahadane": args.vahadane_refs_dir,
    }
    best = {}
    enc_names = {}
    for method in _METHODS:
        print(f"\n--- {method.upper()}  (refs: {refs_dirs[method]}) ---")
        times = []
        for rep in range(args.repeats):
            elapsed, enc_name = _time_method(
                method, refs_dirs[method], base_encoder, df_subset, args,
                coords_dir, device, slide_ext, tmp_dir,
            )
            enc_names[method] = enc_name
            times.append(elapsed)
            print(f"   pass {rep + 1}/{args.repeats}: {elapsed:.2f} s")
        best[method] = min(times)

    # --- Report ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS  (best of {} pass(es))".format(args.repeats))
    print("=" * 70)
    header = f"{'method':<10}{'time (s)':>12}{'slides/s':>12}{'patches/s':>14}{'ms/patch':>12}"
    print(header)
    print("-" * len(header))
    n_slides_done = max(n_with_coords, 1)
    for method in _METHODS:
        t = best[method]
        sps = n_slides_done / t if t > 0 else float("nan")
        pps = n_patches / t if (t > 0 and n_patches) else float("nan")
        mspp = (t * 1000.0 / n_patches) if n_patches else float("nan")
        print(f"{method:<10}{t:>12.2f}{sps:>12.2f}{pps:>14.1f}{mspp:>12.3f}")

    if best["macenko"] > 0:
        factor = best["vahadane"] / best["macenko"]
        faster = "Macenko" if factor >= 1 else "Vahadane"
        print(
            f"\n→ {faster} is {max(factor, 1 / factor):.2f}× faster "
            f"(vahadane/macenko = {factor:.2f})."
        )

    # --- Cleanup ---------------------------------------------------------------
    if not args.keep_outputs:
        for method in _METHODS:
            feat_dir = features_dir(args.job_dir, coords_dir, enc_names[method])
            shutil.rmtree(feat_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\nCleaned up throwaway benchmark outputs "
              "(pass --keep_outputs to retain them).")
    else:
        print("\nKept benchmark feature dirs: "
              + ", ".join(f"features_{enc_names[m]}" for m in _METHODS))

    return 0


if __name__ == "__main__":
    sys.exit(main())
