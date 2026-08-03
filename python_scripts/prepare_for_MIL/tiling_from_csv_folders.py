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
    seg = common + ["--task", "seg", "--segmenter", segmenter, "--gpus", str(gpus)]
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
            reason = ("no year prefix in the slide name" if slide.startswith("?/")
                      else "in no collection, under any extension")
            print(f"  missing: {slide} ({reason})")
        if not found:
            print("  no slides on disk, skipping")
            continue
        results.append((csv_rel, out_dir, found))
    return results


def run_direct(results, root, run, tiling_kwargs):
    """Tile each CSV in one shot, pointing TRIDENT straight at the registry.

    Used for dry runs and --no_staging; `run` gates whether the commands are
    actually executed or only printed.
    """
    for _csv_rel, out_dir, found in results:
        list_csv = write_custom_list(found, os.path.join(out_dir, "slide_list.csv"))
        for command in tiling_commands(list_csv, out_dir, root, **tiling_kwargs):
            print("  " + format_command(command))
            if run:
                subprocess.run(command, check=True)


def run_pipeline(results, root, staging_dir, chunk_size, prefetch, tiling_kwargs):
    """Tile every resolved slide, staging chunks on local disk off the NAS.

    A background thread copies the next chunk into its own staging subdir while
    the current chunk is tiled from local disk; each chunk is deleted once its
    slides are done. The ready queue's bound keeps at most `prefetch` chunks
    staged ahead of the one being tiled, capping local disk use.
    """
    units = []
    for csv_rel, out_dir, found in results:
        for index, chunk in enumerate(chunked(found, chunk_size)):
            units.append(WorkUnit(len(units), csv_rel, out_dir, index, chunk))

    ready = queue.Queue(maxsize=max(1, prefetch))
    done = object()

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
            continue
        dest = os.path.join(staging_dir, str(unit.seq))
        try:
            list_csv = write_custom_list(
                unit.rel_paths,
                os.path.join(unit.out_dir, f"slide_list_{unit.index:04d}.csv"),
            )
            print(f"  {label}: {len(unit.rel_paths)} slides staged -> {dest}")
            for command in tiling_commands(
                list_csv, unit.out_dir, dest, **tiling_kwargs
            ):
                print("    " + format_command(command))
                subprocess.run(command, check=True)
        finally:
            shutil.rmtree(dest, ignore_errors=True)


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
    )

    if args.run and not args.no_staging:
        run_pipeline(
            results,
            args.root,
            args.staging_dir,
            args.chunk_size,
            args.prefetch,
            tiling_kwargs,
        )
    else:
        run_direct(results, args.root, args.run, tiling_kwargs)


if __name__ == "__main__":
    main()
