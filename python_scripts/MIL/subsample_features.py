"""
subsample_features.py — Copy every .npy feature bag keeping at most --max_patches
rows, so a foundation model's features fit in the page cache.

The fusion models read 512 scattered rows per stain per biopsy and epoch. When the
feature folder is larger than RAM (Virchow2: 321 GB, UNI-v2: 205 GB) the page cache
cannot hold it, every epoch reads from disk, and the GPUs wait on I/O. A fixed
random subset of K rows per slide (non-finite rows dropped first) shrinks it to
roughly n_slides x K x D x 2 bytes, read from disk once and from RAM after that.
Training still draws a fresh 512 of the K each epoch; the evenly spaced selection
of validation / inference spreads over the K rows, which keep the slide order.

The kept rows no longer match TRIDENT's patch coordinates, so use these folders
for the fusion models (no coordinates) only, never for the graph models. Compare
foundation models on runs trained and scored on folders made with the same K.

The subset of a slide depends only on its file name, --seed and its number of
finite rows, so a rerun keeps it, and another foundation model keeps the same
patch positions unless their non-finite rows differ (UNI-v2 has a few).
Existing outputs are skipped (an interrupted run can be restarted); a full run
writes ``_complete`` in --dst.

Usage:
    # size of the copy for a few K, nothing written; --labels_csv and --stains
    # keep only the slides the fusion models read
    source stain_config.sh
    python python_scripts/MIL/subsample_features.py \\
        --src WSI/whole_registry/20x_224px_0px_overlap/features_virchow2_npy \\
        --labels_csv "$LABELS_CSV" --stains "${STAINS[@]}" --estimate
    python python_scripts/MIL/subsample_features.py \\
        --src WSI/whole_registry/20x_224px_0px_overlap/features_virchow2_npy \\
        --dst WSI/whole_registry/20x_224px_0px_overlap/features_virchow2_k4096_npy \\
        --labels_csv "$LABELS_CSV" --stains "${STAINS[@]}" --max_patches 4096
    FMS="uni_v2 virchow2" \\
    FEATURES_uni_v2=WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2_k4096_npy \\
    FEATURES_virchow2=WSI/whole_registry/20x_224px_0px_overlap/features_virchow2_k4096_npy \\
        ./train_fusion_models.sh
"""

import argparse
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def subsample(src_path, dst_path, k, seed):
    """Return (name, rows kept, status) with status 'ok', 'skip' or 'error:<msg>'."""
    name = os.path.basename(src_path)
    if os.path.exists(dst_path):
        return name, 0, "skip"
    try:
        x = np.load(src_path)
    except (OSError, ValueError) as e:
        return name, 0, f"error:{e}"
    if x.ndim == 2:
        x = x[np.isfinite(x).all(axis=1)]
        if x.shape[0] > k:
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            x = x[np.sort(rng.choice(x.shape[0], k, replace=False))]
    # Written under a non-.npy name, so an interrupted write is never taken for a bag.
    tmp = dst_path + ".part"
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(x))
    os.replace(tmp, dst_path)
    return name, x.shape[0], "ok"


def estimate(files, label):
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    rows = np.array([s[0] for s in shapes if len(s) == 2])
    row_bytes = np.load(files[0], mmap_mode="r")[:1].nbytes
    print(f"{label}: {len(files)} slides, {rows.sum():,} rows, "
          f"{rows.sum() * row_bytes / 1e9:.0f} GB; median {int(np.median(rows)):,} rows/slide")
    for k in (256, 512, 1024, 2048, 4096):
        kept = np.minimum(rows, k).sum()
        print(f"  --max_patches {k:>6}: {kept * row_bytes / 1e9:6.0f} GB "
              f"({100 * kept / rows.sum():.0f}% of the rows)")


def used_slides(labels_csv, stains):
    """{file name: split} of the slides the fusion models read: listed in the labels
    CSV with one of ``stains`` (entries of STAINS; 'a|b' means stains a and b)."""
    t = pd.read_csv(labels_csv, usecols=["file_name", "stain", "split"])
    if stains:
        names = {n for entry in stains for n in entry.split("|")}
        t = t[t["stain"].isin(names)]
    return dict(zip(t["file_name"].astype(str), t["split"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Folder of .npy feature bags.")
    p.add_argument("--dst", help="Output folder (required unless --estimate).")
    p.add_argument("--max_patches", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--estimate", action="store_true",
                   help="Print the copy's size for several --max_patches and exit.")
    p.add_argument("--labels_csv",
                   help="Keep only the slides of this labels CSV (with --stains, of "
                        "those stains): the slides the fusion models read.")
    p.add_argument("--stains", nargs="+",
                   help="STAINS of stain_config.sh, e.g. 'HE_HES|HE|HES' PAS ...")
    args = p.parse_args()

    files = sorted(os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".npy"))
    split = None
    if args.labels_csv:
        split = used_slides(args.labels_csv, args.stains)
        files = [f for f in files if os.path.basename(f)[:-4] in split]
    if not files:
        p.error(f"no .npy files to copy in {args.src}")
    if args.estimate:
        if split is None:
            estimate(files, "all slides")
            return
        estimate(files, "all used slides (train, val, test)")
        # Read every epoch, so these are what the page cache must hold.
        estimate([f for f in files if split[os.path.basename(f)[:-4]] in ("train", "val")],
                 "train + val (re-read every epoch)")
        return
    if not args.dst:
        p.error("--dst is required unless --estimate")
    os.makedirs(args.dst, exist_ok=True)

    jobs = [(f, os.path.join(args.dst, os.path.basename(f))) for f in files]
    errors, done = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(subsample, s, d, args.max_patches, args.seed) for s, d in jobs]
        for i, fut in enumerate(futures, 1):
            name, _, status = fut.result()
            if status.startswith("error"):
                errors.append(f"{name}: {status[6:]}")
            if i % 500 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} slides")
            done += status != "skip"
    print(f"{done} written, {len(files) - done - len(errors)} already there, {len(errors)} errors")
    for e in errors:
        print("  ", e)
    if not errors:
        open(os.path.join(args.dst, "_complete"), "w").close()


if __name__ == "__main__":
    main()
