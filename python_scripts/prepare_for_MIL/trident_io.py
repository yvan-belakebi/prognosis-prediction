"""
trident_io.py — One place for TRIDENT's on-disk naming conventions.

TRIDENT names the two per-slide artifacts asymmetrically:

    coords:   {job_dir}/{coords_dir}/patches/{slide_id}_patches.h5   (note the _patches suffix)
    features: {job_dir}/{coords_dir}/features_{enc}/{slide_id}.h5     (no suffix)

where  coords_dir = "{mag:g}x_{patch_size}px_{overlap}px_overlap".

Every downstream consumer needs these paths; building the strings inline (and remembering
the _patches suffix, and the exact coords_dir spelling) is error-prone, so the convention
lives here as the single source of truth.

Prefer `discover_coords()` over `coords_path()` when pairing slides to coord files: it
derives slide_id from the files actually on disk, so it cannot silently desync from a labels
CSV whose `file_name` differs from the WSI basename (which otherwise shows up as a confusing
"0 coord files found").
"""

import os

# TRIDENT appends this to every patch-coordinate filename.
TRIDENT_COORD_SUFFIX = "_patches"
_COORD_TAIL = TRIDENT_COORD_SUFFIX + ".h5"


def coords_dir_name(mag, patch_size, overlap):
    """Return TRIDENT's job subdirectory name, e.g. '20x_224px_0px_overlap'."""
    return f"{float(mag):g}x_{int(patch_size)}px_{int(overlap)}px_overlap"


def patches_dir(job_dir, coords_dir):
    """Directory holding the coordinate files for a job."""
    return os.path.join(job_dir, coords_dir, "patches")


def coords_filename(slide_id):
    """Coordinate filename for a slide, including the _patches suffix."""
    return f"{slide_id}{_COORD_TAIL}"


def coords_path(job_dir, coords_dir, slide_id):
    """Full path to a slide's coordinate file (may not exist)."""
    return os.path.join(patches_dir(job_dir, coords_dir), coords_filename(slide_id))


def slide_id_from_coord_file(name):
    """'2013_x_patches.h5' -> '2013_x'.  Returns None if `name` is not a coord file."""
    if name.endswith(_COORD_TAIL):
        return name[: -len(_COORD_TAIL)]
    return None


def discover_coords(patches_dir_path):
    """Scan a patches dir and return {slide_id: coord_path} for every '*_patches.h5'.

    slide_id is derived from the files on disk (suffix stripped), which is robust to a
    labels CSV whose `file_name` does not exactly match the WSI basename. Returns an empty
    dict if the directory is missing.
    """
    found = {}
    if not os.path.isdir(patches_dir_path):
        return found
    for entry in os.scandir(patches_dir_path):
        if entry.is_file():
            slide_id = slide_id_from_coord_file(entry.name)
            if slide_id is not None:
                found[slide_id] = entry.path
    return found


def features_dir(job_dir, coords_dir, enc_name):
    """Directory holding the feature files for an encoder."""
    return os.path.join(job_dir, coords_dir, f"features_{enc_name}")


def feature_path(job_dir, coords_dir, enc_name, slide_id):
    """Full path to a slide's feature file (no _patches suffix)."""
    return os.path.join(features_dir(job_dir, coords_dir, enc_name), f"{slide_id}.h5")


def count_patches(coord_path):
    """Number of patch coordinates in a TRIDENT coord file (0 if missing/unreadable)."""
    import h5py

    try:
        with h5py.File(coord_path, "r") as f:
            return int(f["coords"].shape[0])
    except (OSError, KeyError):
        return 0
