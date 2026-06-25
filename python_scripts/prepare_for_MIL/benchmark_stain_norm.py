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

Slide selection: the subset is drawn at RANDOM from the slides actually present in the
pipeline — the coord files on disk under --job_dir, intersected with the WSIs in --wsi_dir —
never from a CSV list, so it can never pick a slide that isn't tiled. A stain CSV is OPTIONAL
and used only to label the drawn slides (and to spread the draw across stains); slides absent
from it are benchmarked without normalisation.

When the on-disk slides are raw-named but the stain CSV is keyed by anonymized names, pass
--mapping_csv (slide_name_mapping.csv: old_name/new_name). Each raw stem is translated to its
anonymized name purely for the stain lookup — files are still read and written under their raw
names, so nothing on disk needs renaming.

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

from apply_slide_rename import load_mapping  # noqa: E402
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


def _discover_wsi_stems(wsi_dir, slide_ext, search_nested):
    """Return the set of slide stems (basenames) found in wsi_dir for the given extension."""
    stems = set()
    if not os.path.isdir(wsi_dir):
        return stems
    ext = slide_ext.lower()
    if search_nested:
        for _root, _dirs, files in os.walk(wsi_dir):
            for fn in files:
                if fn.lower().endswith(ext):
                    stems.add(os.path.splitext(fn)[0])
    else:
        for e in os.scandir(wsi_dir):
            if e.is_file() and e.name.lower().endswith(ext):
                stems.add(os.path.splitext(e.name)[0])
    return stems


def _build_stain_map(labels_csv, name_col, stain_col):
    """Best-effort {slide_stem: stain} from a CSV (or {} if no CSV).

    The CSV is used only to LABEL slides with a stain, not to choose them. A stem absent
    from the CSV maps to no stain (benchmarked without normalisation). Matching is by exact
    stem, so name_col should hold the same names as the on-disk coord/WSI files.
    """
    if not labels_csv:
        return {}
    df = pd.read_csv(labels_csv)
    if name_col not in df.columns or stain_col not in df.columns:
        raise SystemExit(
            f"--labels_csv must contain '{name_col}' and '{stain_col}' columns "
            f"(got {list(df.columns)}). Use --name_col / --stain_col to override."
        )
    stain_map = {}
    for _, r in df.iterrows():
        stem = str(r[name_col]).strip()
        stain = r[stain_col]
        if stem and pd.notna(stain):
            stain_map[stem] = str(stain)
    return stain_map


def _select_from_disk(pool, stain_map, n_slides, seed):
    """Randomly draw up to n_slides stems from `pool`, spread across stains.

    pool: list of slide stems available on disk (coords present).
    stain_map: {stem: stain}; stems absent are grouped under None (unknown stain).
    Round-robins one slide per stain group at a time (deterministic given seed) so a small
    n_slides still exercises several stains. Returns a list of (stem, stain_or_None).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    by_stain = {}
    for stem in pool:
        by_stain.setdefault(stain_map.get(stem), []).append(stem)
    for stems in by_stain.values():
        rng.shuffle(stems)

    # Deterministic group order: known stains first (sorted), unknown (None) last.
    order = sorted(k for k in by_stain if k is not None)
    if None in by_stain:
        order.append(None)

    chosen = []
    while len(chosen) < n_slides and any(by_stain[k] for k in order):
        for key in order:
            if by_stain[key]:
                chosen.append((by_stain[key].pop(), key))
                if len(chosen) >= n_slides:
                    break
    return chosen


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
    subset,
    args,
    coords_dir,
    device,
    slide_ext,
    tmp_dir,
):
    """Run + time the extraction for one method; returns (elapsed_seconds, enc_name).

    subset: list of (slide_stem, stain_or_None). Slides are grouped by stain so each group
    loads its own reference; a None stain (or a missing reference) runs without normalisation.
    """
    enc_name = f"{base_encoder.enc_name}_bench_{method}"
    file_names = [name for name, _ in subset]

    # Force recompute: drop any features from a previous repeat/run.
    _clear_features(args.job_dir, coords_dir, enc_name, file_names)

    groups = {}
    for name, stain in subset:
        groups.setdefault(stain, []).append(name)

    elapsed = 0.0
    for stain, names in groups.items():
        if stain is None:
            ref_path = None
        else:
            ref_path = os.path.join(refs_dir, f"{_sanitize_stain_name(stain)}.pt")
            if not os.path.isfile(ref_path):
                print(
                    f"  [WARN] {method}: no reference for stain '{stain}' at {ref_path} — "
                    f"this group runs WITHOUT normalisation (timing not comparable)."
                )
                ref_path = None

        label = _sanitize_stain_name(stain) if stain is not None else "unknown"
        rows = [{"wsi": f"{name}{slide_ext}"} for name in names]
        group_csv = os.path.join(tmp_dir, f"{method}_{label}.csv")
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
        default=None,
        help=(
            "Optional CSV used ONLY to assign a stain to each drawn slide (and to spread "
            "the random draw across stains) — NOT to choose the subset. Slides absent from "
            "it are benchmarked without normalisation. Matched by exact stem; see "
            "--name_col / --stain_col."
        ),
    )
    parser.add_argument(
        "--name_col",
        default="file_name",
        help="Column in --labels_csv holding the slide name (matched to on-disk stems).",
    )
    parser.add_argument(
        "--stain_col",
        default="stain",
        help="Column in --labels_csv holding the stain.",
    )
    parser.add_argument(
        "--mapping_csv",
        default=None,
        help=(
            "Optional old_name->new_name CSV (reorganize_wsi_dirs.py --rename output, "
            "slide_name_mapping.csv). When the on-disk slides are raw-named but --labels_csv "
            "is keyed by the anonymized name, this translates each raw stem to its anonymized "
            "name before the stain lookup. Files are still read/written under their raw "
            "names; only the stain assignment is translated."
        ),
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
        help="Number of slides to draw at random from on-disk slides (spread across stains).",
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

    # Draw pool = slides actually in the pipeline (coords on disk), intersected with the
    # WSIs in --wsi_dir when those are enumerable, so we never pick a slide that isn't both
    # tiled and present. Never a CSV list.
    coords_pool = set(discover_coords(patches_dir(args.job_dir, coords_dir)))
    if not coords_pool:
        print(
            f"\n[ERROR] No coord files under {patches_dir(args.job_dir, coords_dir)}. "
            "Run the seg + coords stages first."
        )
        return 1

    wsi_stems = _discover_wsi_stems(args.wsi_dir, slide_ext, args.search_nested)
    if wsi_stems:
        pool = sorted(coords_pool & wsi_stems)
        if not pool:
            print(
                "[WARN] No overlap between coord files and WSIs in --wsi_dir; "
                "drawing from all tiled slides instead."
            )
            pool = sorted(coords_pool)
    else:
        pool = sorted(coords_pool)

    anon_stain = _build_stain_map(args.labels_csv, args.name_col, args.stain_col)
    raw_to_anon = load_mapping(args.mapping_csv) if args.mapping_csv else {}
    # On-disk slides are raw-named while stain info is keyed by the anonymized name.
    # Translate each raw stem -> anonymized name via the mapping (falling back to the raw
    # stem when it is not in the mapping, e.g. already-safe names), then look up its stain.
    stain_by_stem = {}
    for stem in pool:
        stain = anon_stain.get(raw_to_anon.get(stem, stem))
        if stain is not None:
            stain_by_stem[stem] = stain
    subset = _select_from_disk(pool, stain_by_stem, args.n_slides, args.seed)
    if not subset:
        print("\n[ERROR] No slides available to benchmark.")
        return 1

    file_names = [name for name, _ in subset]
    n_patches, found, _missing = _count_patches(args.job_dir, coords_dir, file_names)
    n_with_coords = len(found)
    stains_drawn = sorted({s for _, s in subset if s is not None})
    n_unknown = sum(1 for _, s in subset if s is None)

    print("=" * 70)
    print("STAIN NORMALISATION SPEED BENCHMARK — Macenko vs Vahadane")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Backbone        : {args.backbone}")
    print(f"Coords dir      : {coords_dir}")
    if raw_to_anon:
        print(f"Name mapping    : {len(raw_to_anon)} entries (raw → anonymized)")
    print(f"Pool            : {len(pool)} slide(s) available on disk")
    print(f"Subset          : {len(subset)} slide(s) across {len(stains_drawn)} stain(s)"
          + (f" + {n_unknown} unknown" if n_unknown else ""))
    print(f"Patches         : {n_patches} total")
    print(f"Batch size      : {args.batch_size}  |  repeats: {args.repeats}")
    for name, stain in subset:
        print(f"   - {name}  [{stain if stain is not None else 'unknown stain'}]")

    if not anon_stain:
        print(
            "\n[WARN] No --labels_csv given — all slides run WITHOUT normalisation, so "
            "Macenko and Vahadane will time identically. Pass --labels_csv to enable it."
        )
    elif not stains_drawn:
        hint = (
            "Pass --mapping_csv (slide_name_mapping.csv) to translate the raw on-disk names "
            "to the anonymized names used in --labels_csv."
            if not raw_to_anon
            else f"Check that --mapping_csv new_name values match --labels_csv '{args.name_col}'."
        )
        print(
            f"\n[WARN] None of the drawn slides matched a stain (labels column "
            f"'{args.name_col}'); all run WITHOUT normalisation. {hint}"
        )

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
                method, refs_dirs[method], base_encoder, subset, args,
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
