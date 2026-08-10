"""Apply TRIDENT's --min_tissue_proportion to patches that were already tiled.

Usage:
    # What would be dropped (dry-run):
    python filter_patches_by_tissue.py <job_dir>
    # Actually rewrite the coord files and delete the dumped patch images:
    python filter_patches_by_tissue.py <job_dir> --min_tissue_proportion 0.7 --run

Tiling a run without --min_tissue_proportion keeps every patch that merely
*touches* the tissue mask (TRIDENT's threshold=0 path), so slide edges come with
a fringe of mostly-background patches. This script re-applies the filter after
the fact, without going back to the WSIs: the decision only needs the patch
square and the tissue contours, both of which are already on disk.

It reproduces WSIPatcher._compute_masked exactly -- the contours are simplified
with tolerance = patch_size / 4, and a patch is kept when

    area(patch_square & union(contours)) >= min_tissue_proportion * patch_area

with the square taken in level-0 pixels (patch_size scaled by
level0_magnification / target_magnification, as the patcher rounds it) -- so the
survivors are the patches a run with --min_tissue_proportion would have written.

<job_dir> is walked, so it can be a single TRIDENT job dir or the mirrored tree
that tiling_from_csv_folders.py writes; every

    {job}/{mag}x_{size}px_{overlap}px_overlap/patches/{slide}_patches.h5

found underneath is filtered against {job}/contours_geojson/{slide}.geojson.
The coord file is rewritten in place (its dataset shrunk, its attributes kept)
and stamped with a `min_tissue_proportion` attribute, so re-running skips slides
already filtered at least as strictly. Dumped patch images under
{coords_dir}/patch_images/{slide}/ are deleted for the dropped coordinates,
matched by the x/y in their filenames; the surviving images keep their original
numbering, which is now gappy -- the coordinates in the name, not the index, are
what identifies a patch.

Stale afterwards, and NOT touched here: the {coords_dir}/visualization/ overlays
(they still show the unfiltered grid) and any features_* already extracted from
the old coordinates, which must be re-extracted.
"""

import argparse
import os
import re

import geopandas as gpd
import h5py
from shapely.geometry import box

MIN_TISSUE_PROPORTION = 0.7

# Dumped patch images are named "{index:06d}_x{x}_y{y}.{png,jpg}" by
# WSI.dump_patches, with x/y in level-0 pixels -- the same frame as the coords.
PATCH_IMAGE_RE = re.compile(r"_x(-?\d+)_y(-?\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def find_coord_files(root):
    """Return every TRIDENT coord file under root, sorted.

    A coord file is any "*_patches.h5" sitting in a "patches" dir, which is
    where run_patching_job puts them regardless of how deep the job dir is.
    """
    return sorted(
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(root)
        if os.path.basename(dirpath) == "patches"
        for name in names
        if name.endswith("_patches.h5")
    )


def slide_paths(coord_path):
    """Return (slide, coords_dir, geojson) for a coord file.

    The coord file lives at {job}/{coords_dir}/patches/{slide}_patches.h5 and
    its contours at {job}/contours_geojson/{slide}.geojson, so the job dir is
    three levels up.
    """
    slide = os.path.basename(coord_path)[: -len("_patches.h5")]
    coords_dir = os.path.dirname(os.path.dirname(coord_path))
    job_dir = os.path.dirname(coords_dir)
    geojson = os.path.join(job_dir, "contours_geojson", f"{slide}.geojson")
    return slide, coords_dir, geojson


def patch_size_level0(attrs):
    """Return the patch side in level-0 pixels, as WSIPatcher computes it.

    The patcher works from the magnifications (patch_size * src_mag / dst_mag,
    rounded), not from the `patch_size_level0` attribute, which floors instead
    -- so for a 224px patch at 20x off a 40x slide they can differ by a pixel.
    """
    downsample = float(attrs["level0_magnification"]) / float(
        attrs["target_magnification"]
    )
    return round(int(attrs["patch_size"]) * downsample)


def tissue_proportions(coords, side, contours, patch_size):
    """Return each patch's tissue proportion, i.e. the fraction of its square
    that falls under the contours.

    `side` is the patch side in level-0 pixels and `patch_size` the size at the
    target magnification, which sets the simplification tolerance.
    """
    mask = contours.geometry.simplify(
        tolerance=patch_size / 4, preserve_topology=True
    )
    union = mask.union_all()
    squares = gpd.GeoSeries([box(x, y, x + side, y + side) for x, y in coords])
    return (squares.intersection(union).area / squares.area).to_numpy()


def shrink_coords(coord_path, keep, min_tissue_proportion):
    """Keep only the `keep` coordinates in coord_path, in place.

    The dataset was created resizable, so the survivors are moved to the front
    and it is truncated -- which leaves every attribute untouched. The threshold
    is recorded alongside them so a later run can tell the file is done.
    """
    with h5py.File(coord_path, "r+") as f:
        coords = f["coords"]
        kept = coords[:][keep]
        coords[: len(kept)] = kept
        coords.resize(len(kept), axis=0)
        coords.attrs["min_tissue_proportion"] = min_tissue_proportion


def drop_patch_images(coords_dir, slide, dropped, run):
    """Delete the dumped images of the dropped coordinates. Returns the count.

    Images are matched on the x/y in their filenames rather than on their index,
    which is a position in the old, unfiltered ordering.
    """
    image_dir = os.path.join(coords_dir, "patch_images", slide)
    if not os.path.isdir(image_dir):
        return 0
    dropped = {(int(x), int(y)) for x, y in dropped}
    deleted = 0
    for entry in os.scandir(image_dir):
        match = PATCH_IMAGE_RE.search(entry.name)
        if match and (int(match.group(1)), int(match.group(2))) in dropped:
            if run:
                os.remove(entry.path)
            deleted += 1
    return deleted


def filter_slide(coord_path, min_tissue_proportion, run, patch_images):
    """Filter one slide's coords (and images). Returns (kept, total) or None if
    the slide was skipped."""
    slide, coords_dir, geojson = slide_paths(coord_path)
    with h5py.File(coord_path, "r") as f:
        attrs = dict(f["coords"].attrs)
        coords = f["coords"][:]

    done_at = attrs.get("min_tissue_proportion")
    if done_at is not None and float(done_at) >= min_tissue_proportion:
        print(f"  {slide}: already filtered at {float(done_at):g}, skipping")
        return None
    if len(coords) == 0:
        return 0, 0
    if not os.path.exists(geojson):
        print(f"  {slide}: no contours at {geojson}, skipping")
        return None

    contours = gpd.read_file(geojson)
    if contours.empty:
        # No tissue at all, yet coords exist: the two artifacts disagree, so
        # leave the slide alone rather than silently emptying it.
        print(f"  {slide}: {len(coords)} coords but empty contours, skipping")
        return None

    proportions = tissue_proportions(
        coords, patch_size_level0(attrs), contours, int(attrs["patch_size"])
    )
    # TRIDENT compares inclusively (>=), so 0.7 keeps a patch at exactly 70%.
    keep = proportions >= min_tissue_proportion
    kept, total = int(keep.sum()), len(coords)

    print(f"  {slide}: {kept}/{total} patches kept ({total - kept} dropped)")
    if run:
        shrink_coords(coord_path, keep, min_tissue_proportion)
    if patch_images:
        deleted = drop_patch_images(coords_dir, slide, coords[~keep], run)
        if deleted:
            print(f"    {deleted} patch image(s) {'deleted' if run else 'to delete'}")
    return kept, total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "job_dir", help="TRIDENT job dir, or a tree of them, to filter in place"
    )
    parser.add_argument(
        "--min_tissue_proportion",
        type=float,
        default=MIN_TISSUE_PROPORTION,
        help="minimum proportion of a patch under tissue to keep it, 0.0-1.0 "
        f"(default: {MIN_TISSUE_PROPORTION})",
    )
    parser.add_argument(
        "--patch_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also delete the dumped patch images of dropped patches "
        "(default: on; --no-patch_images to only rewrite the coord files)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="apply the filter (default: report what would be dropped and exit)",
    )
    args = parser.parse_args()

    if not 0 < args.min_tissue_proportion <= 1:
        parser.error("--min_tissue_proportion must be in (0, 1]")

    coord_files = find_coord_files(args.job_dir)
    if not coord_files:
        parser.error(f"no *_patches.h5 found under {args.job_dir}")
    print(f"{len(coord_files)} coord file(s) under {args.job_dir}")
    if not args.run:
        print("Dry run: nothing is modified. Re-run with --run to apply.\n")

    kept_total = dropped_total = 0
    for coord_path in coord_files:
        result = filter_slide(
            coord_path, args.min_tissue_proportion, args.run, args.patch_images
        )
        if result is None:
            continue
        kept, total = result
        kept_total += kept
        dropped_total += total - kept

    print(
        f"\n{kept_total + dropped_total} patches: {kept_total} kept, "
        f"{dropped_total} dropped at min_tissue_proportion="
        f"{args.min_tissue_proportion:g}"
    )
    if args.run:
        print(
            "The visualization/ overlays and any features_* extracted from the "
            "old coordinates are now stale; features must be re-extracted."
        )


if __name__ == "__main__":
    main()
