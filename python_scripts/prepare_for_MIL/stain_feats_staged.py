"""Stage registry WSIs locally, then run per-stain TRIDENT feature extraction.

Feature extraction re-reads WSI pixels at every patch coordinate, so running it
straight off the mounted NAS is dominated by random network reads. This script
drives run_trident_stain_feats.py over slides staged on local disk instead: a
background thread copies the next chunk of slides while the current one is being
encoded, then each chunk is deleted. It is the staging strategy from
tiling_from_csv_folders.py (whose copier, GPU gate and retry helpers it reuses)
applied to the `feat` stage.

Usage:
    # What would run (dry-run):
    python stain_feats_staged.py --wsi_dir /forskning/.../RegistryWSIs \\
        --job_dir WSI/IgA/trident --registry_csv followup_data/derived/renamed/registry_anonymized.csv \\
        --stain_refs_dir stain_refs/IgA
    # Actually extract:
    python stain_feats_staged.py --wsi_dir /forskning/.../RegistryWSIs \\
        --job_dir WSI/IgA/trident --registry_csv followup_data/derived/renamed/registry_anonymized.csv \\
        --stain_refs_dir stain_refs/IgA --staging_dir /data/yvan-files/staging_feats \\
        --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 --run

The slides to process are the ones that ALREADY HAVE TILES: every
`{job_dir}/{coords_dir}/patches/{slide_id}_patches.h5` written by an earlier
TRIDENT coords run. Each slide_id is looked up in --registry_csv (`ANON_name` ->
`Stain`) to get its stain, and the WSI file itself is located by walking
--wsi_dir once. Slides whose stain is absent from the registry are reported and
skipped, never silently extracted unnormalised -- mixing unnormalised features
into a normalised set would quietly corrupt downstream MIL.

Work is chunked per (stain, extension): run_trident_stain_feats.py normalises per
stain and takes a single --slide_ext, so a chunk holding one stain and one
extension keeps each invocation to a single stain group. Slides whose feature .h5
already exists are dropped up front, so a resumed run never re-copies them from
the NAS just for TRIDENT to skip them.
"""

import argparse
import csv
import os
import queue
import shutil
import subprocess
import sys
import threading
from collections import namedtuple

# The staging engine (copier, GPU preflight gate, retrying runner) already lives
# in the tiling script; reuse it rather than growing a second copy.
from tiling_from_csv_folders import (
    child_env,
    chunked,
    copy_into_staging,
    format_command,
    run_with_retries,
    wait_for_gpu,
)
from trident_io import coords_dir_name, discover_coords, patches_dir

FEATS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_trident_stain_feats.py"
)

# Extensions the registry WSIs are scanned in, mirroring tiling_from_csv_folders.
EXTENSIONS = (".svs", ".ndpi")

# One chunk of slides to encode: `seq` gives the chunk its own staging subdir,
# `stain` and `ext` are shared by every slide in it (one stain group, one
# --slide_ext per run_trident_stain_feats.py invocation), and `slides` are
# (slide_id, rel_path) pairs relative to the WSI root.
WorkUnit = namedtuple("WorkUnit", "seq stain ext index slides")


def load_stain_map(registry_csv, name_column="ANON_name", stain_column="Stain"):
    """Return {anon_name: stain} from the registry CSV.

    The registry lists one row per slide; blank stains are treated as missing so
    they surface in the skipped report rather than becoming a bogus group.
    """
    stains = {}
    with open(registry_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for column in (name_column, stain_column):
            if column not in (reader.fieldnames or []):
                raise SystemExit(
                    f"{registry_csv}: missing '{column}' column "
                    f"(found: {', '.join(reader.fieldnames or [])})"
                )
        for row in reader:
            name = (row.get(name_column) or "").strip()
            stain = (row.get(stain_column) or "").strip()
            if name and stain:
                stains[name] = stain
    return stains


def index_wsis(wsi_dir, extensions=EXTENSIONS):
    """Return {slide_id: path relative to wsi_dir} for every WSI under wsi_dir.

    The registry is walked once here so each slide_id taken from the coord files
    can be resolved to a real file without re-hitting the NAS per slide. The
    layout may be biopsy-nested, so the relative path (not just the basename) is
    what gets kept.
    """
    index = {}
    for dirpath, _dirs, names in os.walk(wsi_dir):
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in extensions:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), wsi_dir)
            if stem in index and index[stem] != rel:
                print(f"[WARN] duplicate slide '{stem}' ({index[stem]}, {rel}); "
                      f"keeping {index[stem]}")
                continue
            index[stem] = rel
    return index


def resolve_features_dir(job_dir, coords_dir, enc_name=None):
    """Return the features dir to check for resume, or None if none exists yet.

    The directory is named after the encoder's own `enc_name` (plus any suffix),
    which is only known once the encoder is loaded -- too expensive to do just to
    plan the run. So an existing `features_*` dir is discovered on disk instead;
    --enc_name pins it when several are present.
    """
    parent = os.path.join(job_dir, coords_dir)
    if enc_name:
        return os.path.join(parent, f"features_{enc_name}")
    if not os.path.isdir(parent):
        return None
    candidates = sorted(
        entry.path
        for entry in os.scandir(parent)
        if entry.is_dir() and entry.name.startswith("features_")
    )
    if len(candidates) > 1:
        raise SystemExit(
            "several feature dirs exist ("
            + ", ".join(os.path.basename(c) for c in candidates)
            + "); pass --enc_name to say which one to resume against"
        )
    return candidates[0] if candidates else None


def already_encoded(slide_id, feats_dir):
    """True if this slide's feature .h5 is already on disk."""
    return bool(feats_dir) and os.path.exists(os.path.join(feats_dir, f"{slide_id}.h5"))


def plan_units(tiled, stain_map, wsi_index, feats_dir, chunk_size):
    """Group the tiled slides into per-(stain, extension) chunks to encode.

    Returns (units, skipped) where `skipped` maps a reason to the slide ids
    dropped for it, so the run can report exactly what it is not doing.
    """
    skipped = {"already encoded": [], "no stain in registry": [], "no WSI on disk": []}
    # (stain, ext) -> [(slide_id, rel_path)]
    groups = {}
    for slide_id in sorted(tiled):
        if already_encoded(slide_id, feats_dir):
            skipped["already encoded"].append(slide_id)
            continue
        stain = stain_map.get(slide_id)
        if not stain:
            skipped["no stain in registry"].append(slide_id)
            continue
        rel = wsi_index.get(slide_id)
        if rel is None:
            skipped["no WSI on disk"].append(slide_id)
            continue
        ext = os.path.splitext(rel)[1]
        groups.setdefault((stain, ext), []).append((slide_id, rel))

    units = []
    for (stain, ext), slides in sorted(groups.items()):
        for index, chunk in enumerate(chunked(slides, chunk_size)):
            units.append(WorkUnit(len(units), stain, ext, index, chunk))
    return units, skipped


def write_labels_csv(slides, stain, out_csv):
    """Write the `file_name`,`stain` CSV run_trident_stain_feats.py expects."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "stain"])
        writer.writerows([slide_id, stain] for slide_id, _rel in slides)
    return out_csv


def feats_command(labels_csv, wsi_dir, job_dir, ext, args):
    """Build the run_trident_stain_feats.py command for one staged chunk."""
    command = [
        sys.executable,
        FEATS_SCRIPT,
        "--wsi_dir", wsi_dir,
        "--job_dir", job_dir,
        "--labels_csv", labels_csv,
        "--stain_refs_dir", args.stain_refs_dir,
        "--backbone", args.backbone,
        "--slide_ext", ext,
        "--mag", str(args.mag),
        "--patch_size", str(args.patch_size),
        "--overlap", str(args.overlap),
        "--batch_size", str(args.batch_size),
        "--gpu_index", str(args.gpu_index),
        # Staged copies keep their registry-relative (biopsy-nested) layout, so
        # the feats script has to walk the chunk dir to find them.
        "--search_nested",
    ]
    if args.enc_name_suffix:
        command += ["--enc_name_suffix", args.enc_name_suffix]
    if args.patch_encoder_ckpt_path:
        command += ["--patch_encoder_ckpt_path", args.patch_encoder_ckpt_path]
    if args.max_workers is not None:
        command += ["--max_workers", str(args.max_workers)]
    if args.skip_errors:
        command += ["--skip_errors"]
    return command


def record_failed(failed_log, stain, slide_ids):
    """Append failed slides to failed_log as `{stain}\\t{slide_id}` lines."""
    os.makedirs(os.path.dirname(os.path.abspath(failed_log)), exist_ok=True)
    with open(failed_log, "a", encoding="utf-8") as f:
        for slide_id in slide_ids:
            f.write(f"{stain}\t{slide_id}\n")


def run_pipeline(units, args, coords_dir, failed_log):
    """Encode every chunk, staging slides on local disk off the NAS.

    A background thread copies the next chunk into its own staging subdir while
    the current one is encoded; the ready queue's bound keeps at most --prefetch
    chunks staged ahead, capping local disk use. A chunk whose subprocess fails
    outright is retried, then skipped rather than aborting the run; slides left
    without a feature .h5 afterwards are logged individually.
    """
    ready = queue.Queue(maxsize=max(1, args.prefetch))
    done = object()
    env = child_env()
    failed_chunks = []
    failed_slides = 0

    if os.path.exists(failed_log):
        os.remove(failed_log)

    def copier():
        for unit in units:
            dest = os.path.join(args.staging_dir, str(unit.seq))
            try:
                copy_into_staging([rel for _sid, rel in unit.slides], args.wsi_dir, dest)
            except Exception as error:  # transient NAS/read error: skip the chunk
                shutil.rmtree(dest, ignore_errors=True)
                ready.put((unit, error))
                continue
            ready.put((unit, None))
        ready.put(done)

    threading.Thread(target=copier, daemon=True).start()

    while True:
        item = ready.get()
        if item is done:
            break
        unit, error = item
        label = f"{unit.stain} chunk {unit.index}"
        if error is not None:
            print(f"  {label}: copy failed, skipping ({error})")
            failed_chunks.append(label)
            continue
        dest = os.path.join(args.staging_dir, str(unit.seq))
        try:
            labels_csv = write_labels_csv(
                unit.slides,
                unit.stain,
                os.path.join(
                    args.job_dir, "_staged_feat_lists", f"{unit.seq:04d}.csv"
                ),
            )
            print(f"  {label}: {len(unit.slides)} slides staged -> {dest}")
            if args.gpu_index >= 0:
                wait_for_gpu(args.gpu_index, args.min_free_gib, args.gpu_wait)
            command = feats_command(labels_csv, dest, args.job_dir, unit.ext, args)
            print("    " + format_command(command))
            if not run_with_retries(command, env, args.retries, args.retry_wait):
                print(f"  {label}: giving up, leaving its slides for a re-run")
                failed_chunks.append(label)
                continue
            # The chunk ran to completion; any slide still without its feature
            # .h5 errored inside TRIDENT (--skip_errors) and is logged here.
            feats_dir = resolve_features_dir(args.job_dir, coords_dir, args.enc_name)
            errored = [
                slide_id
                for slide_id, _rel in unit.slides
                if not already_encoded(slide_id, feats_dir)
            ]
            if errored:
                record_failed(failed_log, unit.stain, errored)
                failed_slides += len(errored)
                print(f"  {label}: {len(errored)} slide(s) failed, logged to {failed_log}")
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    if failed_slides:
        print(f"\n{failed_slides} slide(s) failed and were skipped; see {failed_log}")
    if failed_chunks:
        print(f"\n{len(failed_chunks)} chunk(s) failed: {', '.join(failed_chunks)}")
        print("Re-run the same command to retry them (encoded slides are skipped).")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wsi_dir", required=True, help="registry root holding the WSIs")
    parser.add_argument(
        "--job_dir", required=True, help="TRIDENT job dir that already holds the tiles"
    )
    parser.add_argument(
        "--registry_csv",
        required=True,
        help="CSV mapping each slide's base name to its stain (ANON_name, Stain)",
    )
    parser.add_argument(
        "--stain_refs_dir", required=True, help="dir of per-stain .pt references"
    )
    parser.add_argument("--staging_dir", help="local scratch dir; required with --run")
    parser.add_argument("--backbone", default="uni_v2")
    parser.add_argument("--enc_name_suffix", default="")
    parser.add_argument(
        "--enc_name",
        default=None,
        help="encoder dir name to resume against, if several features_* dirs exist",
    )
    parser.add_argument("--mag", type=float, default=20.0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--patch_encoder_ckpt_path", default=None)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="slides staged and encoded per chunk; larger amortizes the encoder "
        "load but uses more local disk",
    )
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry_wait", type=int, default=20)
    parser.add_argument(
        "--min_free_gib",
        type=float,
        default=8.0,
        help="wait for this much free GPU memory before encoding a chunk; "
        "feature extraction needs far more than segmentation",
    )
    parser.add_argument("--gpu_wait", type=int, default=600)
    parser.add_argument(
        "--skip_errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip a slide that errors and keep encoding the rest of its chunk",
    )
    parser.add_argument("--failed_log", default=None)
    parser.add_argument(
        "--run", action="store_true", help="run (default: print the plan and exit)"
    )
    args = parser.parse_args()

    if args.run and not args.staging_dir:
        parser.error("--run needs --staging_dir")
    if args.chunk_size < 1:
        parser.error("--chunk_size must be at least 1")

    coords_dir = coords_dir_name(args.mag, args.patch_size, args.overlap)
    tiled = discover_coords(patches_dir(args.job_dir, coords_dir))
    if not tiled:
        parser.error(
            f"no tiles found under {patches_dir(args.job_dir, coords_dir)} -- "
            "run the coords stage first, or check --mag/--patch_size/--overlap"
        )

    stain_map = load_stain_map(args.registry_csv)
    wsi_index = index_wsis(args.wsi_dir)
    feats_dir = resolve_features_dir(args.job_dir, coords_dir, args.enc_name)
    units, skipped = plan_units(
        tiled, stain_map, wsi_index, feats_dir, args.chunk_size
    )

    print(
        f"{len(tiled)} tiled slide(s) in {coords_dir}; "
        f"{len(stain_map)} stain(s) in the registry; "
        f"{len(wsi_index)} WSI(s) under {args.wsi_dir}"
    )
    for reason, slide_ids in skipped.items():
        if slide_ids:
            sample = ", ".join(slide_ids[:5])
            more = " …" if len(slide_ids) > 5 else ""
            print(f"  skipped, {reason}: {len(slide_ids)} ({sample}{more})")
    total = sum(len(unit.slides) for unit in units)
    print(f"{total} slide(s) to encode in {len(units)} chunk(s)")
    if not units:
        print("nothing to do")
        return

    if not args.run:
        for unit in units:
            print(
                f"  {unit.stain} chunk {unit.index} ({unit.ext}): "
                f"{len(unit.slides)} slides"
            )
        print("\n(dry run; pass --run to execute)")
        return

    failed_log = args.failed_log or os.path.join(args.job_dir, "failed_feat_slides.txt")
    run_pipeline(units, args, coords_dir, failed_log)
    print(f"\nDone. Features -> {os.path.join(args.job_dir, coords_dir)}")


if __name__ == "__main__":
    main()
