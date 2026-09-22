"""Find the slides whose TRIDENT tissue mask holds no polygon.

Usage:
    python find_empty_contours.py <job_dir>
    python find_empty_contours.py <job_dir> --out empty_masks.tsv
    # Feed the mask paths to something else (third column):
    python find_empty_contours.py <job_dir> | cut -f3

TRIDENT's seg pass writes one {stem}.geojson per slide into
{job_dir}/contours_geojson. A slide it found no tissue on still gets a file --
an empty FeatureCollection, logged at the time as "No contour were detected" --
and the coords pass later skips such a slide as "empty_geodataframe", so it
never gets tiles. Nothing on disk marks those slides afterwards: the mask is
present, it is just empty. This lists them.

<job_dir> may be a contours_geojson dir itself or any directory above one;
every contours_geojson below it is scanned, which matches the per-CSV job dirs
tiling_from_csv_folders.py creates. Re-segmenting a slide listed here means
deleting both its .geojson and its contours/{stem}.jpg -- TRIDENT skips a slide
whose contour jpg is in the job dir -- and running the tiling again without
--reuse_segmentation.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

MASK_DIR = "contours_geojson"
# Every geometry in a GeoJSON carries this key, and an empty FeatureCollection
# carries none, so its presence answers the question without parsing the file.
GEOMETRY_MARKER = b'"coordinates"'


def find_mask_dirs(root):
    """Yield every contours_geojson dir at or below root."""
    if os.path.basename(os.path.normpath(root)) == MASK_DIR:
        yield root
        return
    for dirpath, dirnames, _files in os.walk(root):
        if MASK_DIR in dirnames:
            yield os.path.join(dirpath, MASK_DIR)


def find_masks(root):
    """Return every .geojson below root, as (slide stem, path) pairs."""
    masks = []
    for mask_dir in find_mask_dirs(root):
        with os.scandir(mask_dir) as entries:
            masks += [
                (os.path.splitext(entry.name)[0], entry.path)
                for entry in entries
                if entry.is_file() and entry.name.endswith(".geojson")
            ]
    return sorted(masks)


def holds_geometry(path, chunk=1 << 16):
    """True if the file mentions a geometry anywhere.

    Cheaper than parsing, which matters for tens of thousands of files on a
    mount: a mask with tissue hits the marker in its first chunk, and one
    without is a few hundred bytes, so either way this costs about one read.
    """
    tail = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return False
            if GEOMETRY_MARKER in tail + block:
                return True
            tail = block[-len(GEOMETRY_MARKER) :]


def mask_status(path):
    """Return why this mask holds no polygon, or None if it holds one."""
    try:
        if holds_geometry(path):
            return None
        with open(path, encoding="utf-8") as f:
            features = json.load(f).get("features", [])
    except OSError as error:
        return f"unreadable ({error.strerror})"
    except (UnicodeDecodeError, ValueError) as error:  # ValueError: bad JSON
        return f"not valid geojson ({error})"
    # Features without geometry are not what an empty segmentation writes, so
    # they are worth telling apart from it rather than lumping in as "empty".
    return "empty" if not features else "features without geometry"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "job_dir", help="TRIDENT job dir, or a contours_geojson dir directly"
    )
    parser.add_argument(
        "--out", help="write the listing here instead of to stdout (tab-separated)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel reads (default: 16); the scan is latency-bound on a "
        "mounted job dir, so a few threads are worth more than they look",
    )
    args = parser.parse_args()

    masks = find_masks(args.job_dir)
    if not masks:
        parser.error(f"no {MASK_DIR} files found under {args.job_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = pool.map(mask_status, [path for _stem, path in masks])

    lines = [
        f"{stem}\t{status}\t{path}"
        for (stem, path), status in zip(masks, statuses)
        if status is not None
    ]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
    else:
        print("\n".join(lines))
    print(
        f"\n{len(lines)} of {len(masks)} masks hold no polygon"
        + (f" -> {args.out}" if args.out else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
