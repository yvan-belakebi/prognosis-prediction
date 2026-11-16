"""
h5_features_to_npy.py — Copy the 'features' array of every TRIDENT .h5 bag to a
.npy file, for faster training with ``--file_ext .npy``.

The multi-stain dataset reads ~512 scattered patch rows per slide per epoch. From an
.h5 that is a slow fancy-indexed read; from a .npy it is a memory-mapped read that
the OS page cache serves from RAM after the first epoch.

Features are stored as float16 by default (half the disk and page cache; the
training batches are float16 anyway). Values beyond float16's range would become
inf; their count is reported, and such rows are dropped by the dataset like NaN rows.

Only 'features' is copied ('coords' is not used by multistain_MIL.py). Existing
.npy files are skipped, so an interrupted run can be restarted. Unreadable .h5 files
are listed and skipped. A full run ends by writing ``_complete`` in --dst;
train_fusion_models.sh uses the .npy folder only when that marker exists.

Usage:
    python python_scripts/MIL/h5_features_to_npy.py \\
        --src WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2 \\
        --dst WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2_npy
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np


def convert(src_path, dst_path, dtype):
    """Return (name, status) with status 'ok', 'skip', 'inf:<n>' or 'error:<msg>'."""
    name = os.path.basename(src_path)
    if os.path.exists(dst_path):
        return name, "skip"
    try:
        with h5py.File(src_path, "r") as f:
            x = f["features"][()]
    except (OSError, KeyError) as e:
        return name, f"error:{e}"
    with np.errstate(over="ignore"):
        y = x.astype(dtype)
    n_inf = int((np.isinf(y) & np.isfinite(x)).any(axis=-1).sum()) if y.ndim == 2 else 0
    # Write under a non-.npy name, so an interrupted write is never taken for a bag.
    tmp = dst_path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, y)
    os.replace(tmp, dst_path)
    return name, f"inf:{n_inf}" if n_inf else "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Folder of TRIDENT .h5 feature files.")
    p.add_argument("--dst", required=True, help="Output folder for the .npy files.")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    names = sorted(n[:-3] for n in os.listdir(args.src) if n.endswith(".h5"))
    jobs = [(os.path.join(args.src, n + ".h5"), os.path.join(args.dst, n + ".npy"))
            for n in names]
    print(f"{len(jobs)} .h5 files -> {args.dst} ({args.dtype})")

    counts = {"ok": 0, "skip": 0}
    errors, overflows = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(convert, s, d, args.dtype) for s, d in jobs]
        for i, fut in enumerate(futures, 1):
            name, status = fut.result()
            if status.startswith("error:"):
                errors.append(f"{name}: {status[6:]}")
            elif status.startswith("inf:"):
                overflows.append(f"{name}: {status[4:]} rows")
                counts["ok"] += 1
            else:
                counts[status] += 1
            if i % 500 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}", flush=True)

    print(f"converted {counts['ok']}, already present {counts['skip']}, "
          f"unreadable {len(errors)}")
    for e in errors:
        print(f"  unreadable: {e}")
    # Every .h5 has now been converted or found unreadable.
    open(os.path.join(args.dst, "_complete"), "w").close()
    if overflows:
        print(f"{len(overflows)} files have rows that overflow {args.dtype} "
              "(dropped by the dataset like NaN rows):")
        for o in overflows:
            print(f"  {o}")


if __name__ == "__main__":
    main()
