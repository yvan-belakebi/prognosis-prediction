"""Tile the registry WSIs listed across a tree of slide-list CSVs, using TRIDENT.

Usage:
    # What would run (dry-run):
    python tiling_for_hrafn.py <csv_dir> --job_dir WSI/hrafn/trident
    # Actually run segmentation + patch coordinates:
    python tiling_for_hrafn.py <csv_dir> --job_dir WSI/hrafn/trident --run

<csv_dir> is a folder of subfolders, each holding slide-list CSVs with header
columns `wsi_anon_name`, `year` and `lab_name` (see file_for_hrafn.csv). Every
CSV found below <csv_dir> is read, and each slide is looked up in the registry
under {root}/{collection}/{year}/{wsi_anon_name}{.svs,.ndpi}. The CSVs record
neither the collection a slide lives in nor the format it was scanned in, so
every combination is tried and the first one on disk wins.

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
import subprocess
import sys

REGISTRY_ROOT = "/forskning/hbe/2023-517496/RegistryWSIs"
COLLECTIONS = ("The Norwegian Kidney Biopsy Registry", "Kidney biopsies")
EXTENSIONS = (".svs", ".ndpi")

TRIDENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external_repositories",
    "TRIDENT-main",
    "run_batch_of_slides.py",
)


def year_dir(value):
    """Return the "{year}_anon" dir for a CSV year cell, or None if unusable.

    year is written as a float ("2014.0") but is a directory name. Cells are
    occasionally blank, which is a slide we cannot place rather than a fatal
    error.
    """
    try:
        return f"{int(float(value))}_anon"
    except (TypeError, ValueError):
        return None


def slide_candidates(csv_path, collections=COLLECTIONS, extensions=EXTENSIONS):
    """Return [(slide, [candidate rel paths])] for every slide listed in csv_path.

    `slide` is the collection- and extension-independent "{year}_anon/{name}"
    stem, so a slide keeps one identity across the places it might be stored. A
    row with no usable year gets no candidates, so it reports as unresolved
    instead of derailing the whole CSV.
    """
    slides = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("wsi_anon_name") or "").strip()
            if not name:
                continue
            year = year_dir(row.get("year"))
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
    ]
    return [seg, coords]


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
        "--run",
        action="store_true",
        help="run TRIDENT (default: print the commands and exit)",
    )
    args = parser.parse_args()

    csv_paths = find_csvs(args.csv_dir)
    if not csv_paths:
        parser.error(f"no CSVs found under {args.csv_dir}")

    total_found = 0
    for csv_path in csv_paths:
        found, missing = resolve_slides(csv_path, args.root)
        total_found += len(found)
        out_dir = job_dir_for(csv_path, args.csv_dir, args.job_dir)
        print(
            f"\n{os.path.relpath(csv_path, args.csv_dir)}: "
            f"{len(found)} slides found, {len(missing)} missing -> {out_dir}"
        )
        for slide in missing:
            reason = ("no year in CSV" if slide.startswith("?/")
                      else "in no collection, under any extension")
            print(f"  missing: {slide} ({reason})")
        if not found:
            print("  no slides on disk, skipping")
            continue

        list_csv = write_custom_list(found, os.path.join(out_dir, "slide_list.csv"))
        commands = tiling_commands(
            list_csv,
            out_dir,
            args.root,
            args.mag,
            args.patch_size,
            args.overlap,
            args.segmenter,
            args.gpus,
        )
        for command in commands:
            print(
                "  "
                + " ".join(f'"{part}"' if " " in part else part for part in command)
            )
            if args.run:
                subprocess.run(command, check=True)

    if not total_found:
        parser.error(
            f"no slides resolved under {args.root} -- is the registry mounted?"
        )


if __name__ == "__main__":
    main()
