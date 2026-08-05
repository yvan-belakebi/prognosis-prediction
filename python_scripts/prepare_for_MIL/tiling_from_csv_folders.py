"""Tile the registry WSIs listed across a tree of slide-list CSVs, using TRIDENT.

Usage:
    # What would run (dry-run):
    python tiling_from_csv_folders.py <csv_dir> --job_dir WSI/hrafn/trident
    # Actually run segmentation + patch coordinates, keeping only patches at
    # least 70% under the tissue mask:
    python tiling_from_csv_folders.py <csv_dir> --job_dir WSI/hrafn/trident \
        --staging_dir /local/scratch/staging --min_tissue_proportion 0.7 --run
    # Bypass staging and read the registry directly (the old, NAS-bound path):
    python tiling_from_csv_folders.py <csv_dir> --job_dir WSI/hrafn/trident \
        --no_staging --run

Reading the registry WSIs straight off the mounted NAS makes TRIDENT's random
per-tile reads slow. With --staging_dir, slides are copied to local disk in
chunks and tiled from there, then deleted; a background thread stages the next
chunk while the current one is tiled, so the NAS copy overlaps with the GPU
work. --chunk_size and --prefetch bound how much local disk is used at once
(roughly (prefetch + 1) chunks). A chunk is tiled by a single TRIDENT
invocation, so a larger chunk also amortizes TRIDENT's per-run model loading.

<csv_dir> is a folder of subfolders, each holding slide-list CSVs with header
columns `wsi_anon_name`, `year` and `lab_name` (see file_for_hrafn.csv). Every
CSV found below <csv_dir> is read, and each slide is looked up in the registry
under {root}/{collection}/{year}_anon/{wsi_anon_name}{.svs,.ndpi}, where {year}
is the prefix of the slide name ("{year}_{id}_ANON"). The CSV's own `year`
column is ignored: it disagrees with the name prefix for a good share of slides,
and the prefix is the one that matches the registry. The CSVs record neither the
collection a slide lives in nor the format it was scanned in, so every
combination is tried and the first one on disk wins.

Each CSV is tiled into its own job dir, mirroring the CSV tree under --job_dir:
<csv_dir>/labA/a.csv is tiled into <job_dir>/labA/a/. A slide listed in two CSVs
is therefore tiled once per CSV. Slides are deduplicated within a CSV, and the
survivors are written to that job dir's TRIDENT `--custom_list_of_wsis` CSV,
whose `wsi` column holds paths relative to the registry root (passed to TRIDENT
as `--wsi_dir`). Slides missing from disk are reported and dropped, since
TRIDENT errors out on a list entry it cannot find.
"""

import argparse
import csv
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import namedtuple

REGISTRY_ROOT = "/forskning/hbe/2023-517496/RegistryWSIs"
COLLECTIONS = ("The Norwegian Kidney Biopsy Registry", "Kidney biopsies")
EXTENSIONS = (".svs", ".ndpi")
# Keep a patch only if at least this much of it sits under the segmentation
# mask. TRIDENT compares inclusively (>=), so 0.7 keeps a patch at exactly 70%.
MIN_TISSUE_PROPORTION = 0.7

TRIDENT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external_repositories",
    "TRIDENT-main",
    "run_batch_of_slides.py",
)


def year_dir(name):
    """Return the "{year}_anon" dir for a slide name, or None if it has no year.

    Slide names are "{year}_{id}_ANON". The year prefix -- not the CSV's `year`
    column, which disagrees with it for a good share of slides -- is what
    matches the registry layout.
    """
    year = name.split("_", 1)[0]
    return f"{year}_anon" if year.isdigit() else None


def slide_candidates(csv_path, collections=COLLECTIONS, extensions=EXTENSIONS):
    """Return [(slide, [candidate rel paths])] for every slide listed in csv_path.

    `slide` is the collection- and extension-independent "{year}_anon/{name}"
    stem, so a slide keeps one identity across the places it might be stored. A
    row whose name carries no year gets no candidates, so it reports as
    unresolved instead of derailing the whole CSV.
    """
    slides = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("wsi_anon_name") or "").strip()
            if not name:
                continue
            year = year_dir(name)
            if year is None:
                slides.append((f"?/{name}", []))
                continue
            slides.append(
                (
                    f"{year}/{name}",
                    [
                        f"{collection}/{year}/{name}{extension}"
                        for collection in collections
                        for extension in extensions
                    ],
                )
            )
    return slides


def candidate_rel_paths(csv_path, collections=COLLECTIONS, extensions=EXTENSIONS):
    """Return the candidate registry paths, relative to the root, for one CSV."""
    return [
        rel
        for _, rel_paths in slide_candidates(csv_path, collections, extensions)
        for rel in rel_paths
    ]


def candidate_paths(
    csv_path, root=REGISTRY_ROOT, collections=COLLECTIONS, extensions=EXTENSIONS
):
    """Return the candidate registry paths for every slide listed in csv_path."""
    return [
        f"{root}/{rel}"
        for rel in candidate_rel_paths(csv_path, collections, extensions)
    ]


def find_csvs(csv_dir):
    """Return every CSV below csv_dir, sorted."""
    return sorted(
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(csv_dir)
        for name in names
        if name.lower().endswith(".csv")
    )


def resolve_slides(
    csv_path, root=REGISTRY_ROOT, collections=COLLECTIONS, extensions=EXTENSIONS
):
    """Resolve the slides listed in csv_path against the registry.

    Returns (found, missing): `found` holds the deduplicated relative paths of
    slides present on disk, `missing` the "{year}/{name}" stems that turned up
    in no collection under any extension.
    """
    # slide -> the relative path that exists on disk, or None if none does.
    resolved = {}
    for slide, rel_paths in slide_candidates(csv_path, collections, extensions):
        if resolved.get(slide) is not None:
            continue
        resolved[slide] = next(
            (rel for rel in rel_paths if os.path.exists(os.path.join(root, rel))), None
        )
    found = [rel for rel in resolved.values() if rel is not None]
    missing = [slide for slide, rel in resolved.items() if rel is None]
    return found, missing


def job_dir_for(csv_path, csv_dir, job_dir):
    """Return the job dir for csv_path, mirroring its place in the csv_dir tree."""
    relative = os.path.relpath(csv_path, csv_dir)
    return os.path.join(job_dir, os.path.splitext(relative)[0])


def write_custom_list(rel_paths, out_csv):
    """Write rel_paths as a TRIDENT --custom_list_of_wsis CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi"])
        writer.writerows([rel] for rel in rel_paths)
    return out_csv


def tiling_commands(
    list_csv,
    job_dir,
    root=REGISTRY_ROOT,
    mag=20,
    patch_size=224,
    overlap=0,
    segmenter="hest",
    gpus=0,
    min_tissue_proportion=MIN_TISSUE_PROPORTION,
    dump_patches=True,
    seg_batch_size=None,
    skip_errors=True,
    clear_dead_locks=True,
):
    """Return the TRIDENT seg and coords commands for the slides in list_csv."""
    common = [
        sys.executable,
        TRIDENT,
        "--wsi_dir",
        root,
        "--job_dir",
        job_dir,
        "--custom_list_of_wsis",
        list_csv,
    ]
    if skip_errors:
        # Let TRIDENT skip a slide that errors (recording it in its own
        # _logs_{segmentation,coords}.txt) and carry on with the rest of the
        # chunk, instead of aborting the whole invocation on the first failure.
        common += ["--skip_errors"]
    if clear_dead_locks:
        # A slide that errors under --skip_errors leaves its {name}.jpg.lock
        # behind (TRIDENT only removes it on KeyboardInterrupt), and a later run
        # then skips that slide as "locked by another worker" -- producing only
        # .jpg.lock files and no patches. Clear such orphaned locks from crashed
        # runs first; locks held by a live PID are kept, so this is safe here
        # where only one TRIDENT runs at a time.
        common += ["--clear_dead_locks"]
    seg = common + ["--task", "seg", "--segmenter", segmenter, "--gpus", str(gpus)]
    if seg_batch_size is not None:
        seg += ["--seg_batch_size", str(seg_batch_size)]
    coords = common + [
        "--task",
        "coords",
        "--mag",
        str(mag),
        "--patch_size",
        str(patch_size),
        "--overlap",
        str(overlap),
        "--min_tissue_proportion",
        str(min_tissue_proportion),
    ]
    if dump_patches:
        coords.append("--dump_patches")
    return [seg, coords]


# One chunk of slides to tile: `seq` is a run-wide id used to give the chunk its
# own staging subdir (so a slide listed in two CSVs never has one chunk's copy
# clobber another's), `index` is the chunk's position within its CSV (used to
# name the chunk's TRIDENT list), and `rel_paths` are its registry-relative WSIs.
WorkUnit = namedtuple("WorkUnit", "seq csv_rel out_dir index rel_paths")


def format_command(command):
    """Render a command list for printing, quoting the parts that hold spaces."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def chunked(seq, size):
    """Yield successive size-length slices of seq."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def copy_into_staging(rel_paths, root, dest_root):
    """Copy each registry-relative slide into dest_root, keeping its rel path."""
    for rel in rel_paths:
        dst = os.path.join(dest_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(root, rel), dst)


def resolve_all(csv_paths, csv_dir, job_dir, root):
    """Resolve every CSV against the registry, print the per-CSV report, and
    return [(csv_rel, out_dir, found)] for the CSVs with slides on disk."""
    results = []
    for csv_path in csv_paths:
        found, missing = resolve_slides(csv_path, root)
        out_dir = job_dir_for(csv_path, csv_dir, job_dir)
        csv_rel = os.path.relpath(csv_path, csv_dir)
        print(
            f"\n{csv_rel}: {len(found)} slides found, {len(missing)} missing "
            f"-> {out_dir}"
        )
        for slide in missing:
            reason = (
                "no year prefix in the slide name"
                if slide.startswith("?/")
                else "in no collection, under any extension"
            )
            print(f"  missing: {slide} ({reason})")
        if not found:
            print("  no slides on disk, skipping")
            continue
        results.append((csv_rel, out_dir, found))
    return results


def child_env():
    """Environment for TRIDENT subprocesses.

    Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (unless already set) to
    reduce CUDA fragmentation; retries handle the transient shared-GPU contention
    that fragmentation alone does not explain.
    """
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def gpu_free_gib(gpu):
    """Free memory (GiB) on GPU `gpu` via nvidia-smi, or None if it can't be read
    (nvidia-smi missing, a CPU run, etc.), in which case callers skip the gate."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def wait_for_gpu(gpu, min_free_gib, max_wait, poll=15):
    """Block until GPU `gpu` has `min_free_gib` free, or `max_wait` seconds pass.

    Guards against launching a chunk into a momentarily-full shared vGPU (where
    it would OOM, or -- under --skip_errors -- silently mass-skip). If nvidia-smi
    is unreadable the gate is a no-op; if the GPU never frees up within max_wait
    it proceeds anyway, leaving the retry/skip paths to handle the fallout.
    """
    waited = 0
    while True:
        free = gpu_free_gib(gpu)
        if free is None or free >= min_free_gib:
            return
        if waited >= max_wait:
            print(
                f"    GPU {gpu}: only {free:.1f} GiB free after {waited}s, "
                f"proceeding anyway"
            )
            return
        print(
            f"    GPU {gpu}: {free:.1f} GiB free (< {min_free_gib}), "
            f"waiting {poll}s for it to clear"
        )
        time.sleep(poll)
        waited += poll


def run_with_retries(command, env, retries, retry_wait):
    """Run command, retrying on a non-zero exit. Return True if it ever succeeds.

    A retried TRIDENT call re-scans the job dir and skips slides it already
    finished, so retrying a partly-done chunk only redoes the slide it died on.
    """
    for attempt in range(retries + 1):
        try:
            subprocess.run(command, check=True, env=env)
            return True
        except subprocess.CalledProcessError as error:
            if attempt < retries:
                print(
                    f"    failed (attempt {attempt + 1}/{retries + 1}), "
                    f"retrying in {retry_wait}s: {error}"
                )
                time.sleep(retry_wait)
            else:
                print(f"    failed after {retries + 1} attempts: {error}")
    return False


def run_direct(results, root, run, tiling_kwargs):
    """Tile each CSV in one shot, pointing TRIDENT straight at the registry.

    Used for dry runs and --no_staging; `run` gates whether the commands are
    actually executed or only printed.
    """
    env = child_env()
    for _csv_rel, out_dir, found in results:
        list_csv = write_custom_list(found, os.path.join(out_dir, "slide_list.csv"))
        for command in tiling_commands(list_csv, out_dir, root, **tiling_kwargs):
            print("  " + format_command(command))
            if run:
                subprocess.run(command, check=True, env=env)


def coords_dirname(mag, patch_size, overlap):
    """The subdir TRIDENT writes patch coords into, matching its own naming
    (`{mag:g}x_{patch}px_{overlap}px_overlap`, e.g. `5x_224px_0px_overlap`)."""
    return f"{float(mag):g}x_{patch_size}px_{overlap}px_overlap"


def already_tiled(rel_path, out_dir, coords_subdir):
    """True if this slide's patch-coords .h5 already exists under out_dir, i.e.
    it was tiled on an earlier run and TRIDENT would only skip it again."""
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    return os.path.exists(
        os.path.join(out_dir, coords_subdir, "patches", f"{stem}_patches.h5")
    )


def record_failed_slides(failed_log, csv_rel, rel_paths):
    """Append the failed slides to failed_log as `{csv_rel}\\t{rel_path}` lines.

    A zero-tissue slide still writes an (empty) _patches.h5, so a slide left
    without one after a --skip_errors run genuinely errored -- that absence is
    what marks it failed here.
    """
    os.makedirs(os.path.dirname(os.path.abspath(failed_log)), exist_ok=True)
    with open(failed_log, "a", encoding="utf-8") as f:
        for rel in rel_paths:
            f.write(f"{csv_rel}\t{rel}\n")


def run_pipeline(
    results,
    root,
    staging_dir,
    chunk_size,
    prefetch,
    retries,
    retry_wait,
    min_free_gib,
    gpu_wait,
    failed_log,
    tiling_kwargs,
):
    """Tile every resolved slide, staging chunks on local disk off the NAS.

    A background thread copies the next chunk into its own staging subdir while
    the current chunk is tiled from local disk; each chunk is deleted once its
    slides are done. The ready queue's bound keeps at most `prefetch` chunks
    staged ahead of the one being tiled, capping local disk use.

    Slides already tiled on an earlier run (their patch-coords .h5 is on disk)
    are dropped up front, so a resumed run never copies them from the NAS just
    for TRIDENT to skip them.

    An individual slide that errors is skipped by TRIDENT (via --skip_errors)
    and its name written to failed_log, while the rest of the chunk is still
    tiled. A chunk that fails outright (e.g. a whole-process crash that
    --skip_errors cannot catch, like a transient shared-GPU OOM) is retried in
    place, then skipped rather than aborting the whole run; such chunks are
    reported at the end and the run exits non-zero so a re-run can mop them up
    (finished slides are skipped by TRIDENT's own resume).
    """
    coords_subdir = coords_dirname(
        tiling_kwargs["mag"], tiling_kwargs["patch_size"], tiling_kwargs["overlap"]
    )
    units = []
    skipped = 0
    for csv_rel, out_dir, found in results:
        pending = [
            rel for rel in found if not already_tiled(rel, out_dir, coords_subdir)
        ]
        skipped += len(found) - len(pending)
        for index, chunk in enumerate(chunked(pending, chunk_size)):
            units.append(WorkUnit(len(units), csv_rel, out_dir, index, chunk))
    if skipped:
        print(f"\nAlready tiled on a previous run, not re-copying: {skipped} slides.")

    # Start each run with a fresh failed-slides log; resume re-attempts the
    # slides listed here (they still lack a .h5), so a stale list would mislead.
    if os.path.exists(failed_log):
        os.remove(failed_log)

    ready = queue.Queue(maxsize=max(1, prefetch))
    done = object()
    env = child_env()
    gpu = tiling_kwargs["gpus"]  # index the preflight gate polls; < 0 means CPU
    failed = []
    failed_slides = 0

    def copier():
        for unit in units:
            dest = os.path.join(staging_dir, str(unit.seq))
            try:
                copy_into_staging(unit.rel_paths, root, dest)
            except Exception as error:  # transient NAS/read error: skip the chunk
                shutil.rmtree(dest, ignore_errors=True)
                ready.put((unit, error))
                continue
            ready.put((unit, None))
        ready.put(done)

    thread = threading.Thread(target=copier, daemon=True)
    thread.start()

    while True:
        item = ready.get()
        if item is done:
            break
        unit, error = item
        label = f"{unit.csv_rel} chunk {unit.index}"
        if error is not None:
            print(f"  {label}: copy failed, skipping ({error})")
            failed.append(label)
            continue
        dest = os.path.join(staging_dir, str(unit.seq))
        try:
            list_csv = write_custom_list(
                unit.rel_paths,
                os.path.join(unit.out_dir, f"slide_list_{unit.index:04d}.csv"),
            )
            print(f"  {label}: {len(unit.rel_paths)} slides staged -> {dest}")
            if gpu >= 0:
                wait_for_gpu(gpu, min_free_gib, gpu_wait)
            chunk_ok = True
            for command in tiling_commands(
                list_csv, unit.out_dir, dest, **tiling_kwargs
            ):
                print("    " + format_command(command))
                if not run_with_retries(command, env, retries, retry_wait):
                    print(f"  {label}: giving up, leaving its slides for a re-run")
                    failed.append(label)
                    chunk_ok = False
                    break
            if chunk_ok:
                # The chunk ran to completion; any slide still without its .h5
                # was skipped by --skip_errors. Log it, keep the rest.
                errored = [
                    rel
                    for rel in unit.rel_paths
                    if not already_tiled(rel, unit.out_dir, coords_subdir)
                ]
                if errored:
                    record_failed_slides(failed_log, unit.csv_rel, errored)
                    failed_slides += len(errored)
                    print(
                        f"  {label}: {len(errored)} slide(s) failed and were "
                        f"skipped, logged to {failed_log}"
                    )
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    if failed_slides:
        print(f"\n{failed_slides} slide(s) failed and were skipped; see {failed_log}")
    if failed:
        print(f"\n{len(failed)} chunk(s) failed: {', '.join(failed)}")
        print("Re-run the same command to retry them (finished slides are skipped).")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csv_dir", help="folder of subfolders containing slide-list CSVs"
    )
    parser.add_argument(
        "--job_dir",
        required=True,
        help="TRIDENT output root; the csv_dir " "tree is mirrored underneath it",
    )
    parser.add_argument(
        "--root", default=REGISTRY_ROOT, help="registry root (TRIDENT --wsi_dir)"
    )
    parser.add_argument("--mag", type=float, default=20)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--segmenter", default="hest")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument(
        "--min_tissue_proportion",
        type=float,
        default=MIN_TISSUE_PROPORTION,
        help="minimum proportion of a patch under tissue to keep it, 0.0-1.0 "
        f"(default: {MIN_TISSUE_PROPORTION}); 0 keeps every patch",
    )
    parser.add_argument(
        "--dump_patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write the patch images to disk, not just their coordinates "
        "(default: on; --no-dump_patches for coordinates only)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run TRIDENT (default: print the commands and exit)",
    )
    parser.add_argument(
        "--staging_dir",
        help="local scratch dir; slides are copied here in chunks before tiling "
        "and deleted after, keeping the slow NAS off the tiling path. Required "
        "with --run unless --no_staging is given.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="slides copied and tiled per chunk (default: 8). Larger amortizes "
        "TRIDENT's per-run model loading but uses more local disk.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="chunks staged ahead of the one being tiled (default: 1); local "
        "disk holds up to ~(prefetch + 1) chunks at once.",
    )
    parser.add_argument(
        "--no_staging",
        action="store_true",
        help="read slides straight from the registry instead of staging them "
        "locally (the old, NAS-bound behavior)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="times to retry a chunk that fails, e.g. on a transient shared-GPU "
        "OOM, before skipping it (default: 1); a retry reuses the staged slides "
        "and skips ones already finished",
    )
    parser.add_argument(
        "--retry_wait",
        type=int,
        default=20,
        help="seconds to wait before retrying a failed chunk (default: 20), "
        "giving transient GPU contention time to clear",
    )
    parser.add_argument(
        "--seg_batch_size",
        type=int,
        default=None,
        help="TRIDENT segmentation batch size (default: TRIDENT's own, 64); "
        "lower it to shrink the GPU footprint on a busy shared card",
    )
    parser.add_argument(
        "--min_free_gib",
        type=float,
        default=4.0,
        help="before tiling a chunk, wait until the GPU has at least this many "
        "GiB free (default: 4.0), so a chunk never launches into a momentarily-"
        "full shared vGPU; 0 disables the wait",
    )
    parser.add_argument(
        "--gpu_wait",
        type=int,
        default=600,
        help="max seconds to wait for the GPU to free up before tiling a chunk "
        "(default: 600); after this it proceeds anyway",
    )
    parser.add_argument(
        "--skip_errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip a slide that errors and keep tiling the rest of its chunk, "
        "logging the failed slide names (default: on; --no-skip_errors stops "
        "the chunk on the first slide error)",
    )
    parser.add_argument(
        "--clear_dead_locks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="clear orphaned TRIDENT .lock files from earlier crashed/OOM'd "
        "runs before tiling (default: on); without this, slides left locked by "
        "a dead run are skipped forever and yield only .jpg.lock files",
    )
    parser.add_argument(
        "--failed_log",
        help="file to append failed slide names to, as `csv\\tslide` lines "
        "(default: <job_dir>/failed_slides.txt); rewritten fresh each run",
    )
    args = parser.parse_args()

    if args.run and not args.no_staging and not args.staging_dir:
        parser.error(
            "--run needs --staging_dir (or pass --no_staging to read the "
            "registry directly)"
        )
    if args.chunk_size < 1:
        parser.error("--chunk_size must be at least 1")

    csv_paths = find_csvs(args.csv_dir)
    if not csv_paths:
        parser.error(f"no CSVs found under {args.csv_dir}")

    results = resolve_all(csv_paths, args.csv_dir, args.job_dir, args.root)
    total_found = sum(len(found) for _, _, found in results)
    if not total_found:
        parser.error(
            f"no slides resolved under {args.root} -- is the registry mounted?"
        )

    tiling_kwargs = dict(
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        segmenter=args.segmenter,
        gpus=args.gpus,
        min_tissue_proportion=args.min_tissue_proportion,
        dump_patches=args.dump_patches,
        seg_batch_size=args.seg_batch_size,
        skip_errors=args.skip_errors,
        clear_dead_locks=args.clear_dead_locks,
    )

    if args.run and not args.no_staging:
        failed_log = args.failed_log or os.path.join(args.job_dir, "failed_slides.txt")
        run_pipeline(
            results,
            args.root,
            args.staging_dir,
            args.chunk_size,
            args.prefetch,
            args.retries,
            args.retry_wait,
            args.min_free_gib,
            args.gpu_wait,
            failed_log,
            tiling_kwargs,
        )
    else:
        run_direct(results, args.root, args.run, tiling_kwargs)


if __name__ == "__main__":
    main()
