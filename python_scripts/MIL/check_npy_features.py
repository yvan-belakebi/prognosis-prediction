"""
check_npy_features.py — Check the .npy feature copies before deleting the .h5 originals.

For every slide:
  - the .npy has a coordinate file in --coords (TRIDENT's {slide}_patches.h5) with
    one row per feature row, since the graph models pair them row by row;
  - with --h5 (the original feature folder): every .h5 has a .npy, the .npy equals
    the .h5 features rounded to float16 (NaN where the .h5 has NaN), and the .h5's
    own coords equal those of the patch file, row for row.

Empty bags (no patches) are skipped: training drops them anyway. Exits with
status 1 and lists the slides when anything is wrong.

Usage:
    python python_scripts/MIL/check_npy_features.py \\
        --features WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2_npy \\
        --coords   WSI/whole_registry/20x_224px_0px_overlap/patches \\
        --h5       WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np


def coords_file(coords, name):
    """TRIDENT's {name}_patches.h5, or {name}.h5 (same rule as mil_utils)."""
    path = os.path.join(coords, name + ".h5")
    return path if os.path.exists(path) else os.path.join(coords, name + "_patches.h5")


def check(name, features, coords, h5):
    """Return a problem description, or None when the slide is fine."""
    x = np.load(os.path.join(features, name + ".npy"), mmap_mode="r")
    if x.ndim < 2 or x.shape[0] == 0:
        return None
    cpath = coords_file(coords, name)
    if not os.path.exists(cpath):
        return f"no coordinate file in {coords}"
    with h5py.File(cpath, "r") as f:
        c = f["coords"][:]
    if c.shape[0] != x.shape[0]:
        return f"{x.shape[0]} feature rows but {c.shape[0]} coordinates"
    if h5 is None:
        return None
    with h5py.File(os.path.join(h5, name + ".h5"), "r") as f:
        ref = f["features"][:]
        ref_c = f["coords"][:] if "coords" in f else None
    if ref.shape != x.shape:
        return f".npy shape {x.shape} != .h5 shape {ref.shape}"
    with np.errstate(over="ignore"):
        ref16 = ref.astype(np.float16)
    same = (ref16 == x) | (np.isnan(ref16) & np.isnan(x))
    if not same.all():
        return f"{int((~same.all(axis=1)).sum())} feature rows differ from the .h5"
    if ref_c is not None and not np.array_equal(ref_c, c):
        return "coordinates differ between the .h5 and the patch file"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="Folder of .npy features.")
    p.add_argument("--coords", required=True, help="TRIDENT patches/ folder.")
    p.add_argument("--h5", default=None, help="Original .h5 feature folder (optional).")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    names = sorted(n[:-4] for n in os.listdir(args.features) if n.endswith(".npy"))
    problems = []
    if args.h5:
        h5_names = {n[:-3] for n in os.listdir(args.h5) if n.endswith(".h5")}
        problems += [(n, "has a .h5 but no .npy") for n in sorted(h5_names - set(names))]
        names = [n for n in names if n in h5_names]
    print(f"checking {len(names)} slides" + (" against the .h5 files" if args.h5 else ""))

    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(check, n, args.features, args.coords, args.h5) for n in names]
        for i, (n, fut) in enumerate(zip(names, futures), 1):
            try:
                msg = fut.result()
            except Exception as e:  # unreadable file
                msg = f"error: {e}"
            if msg:
                problems.append((n, msg))
            if i % 1000 == 0 or i == len(names):
                print(f"  {i}/{len(names)}", flush=True)

    if not problems:
        print("OK — every slide has matching .npy features and coordinates.")
        return
    print(f"{len(problems)} problem(s):")
    for n, msg in problems:
        print(f"  {n}: {msg}")
    sys.exit(1)


if __name__ == "__main__":
    main()
