"""
sync_dirs.py — Remove files from dir B that have no matching name in dir A.

Only filenames (without extension) are compared, so A/foo.h5 keeps B/foo.npy.
Use --dry_run to preview deletions before committing.

Usage:
    python sync_dirs.py --dir_a WSI/IgA/UNI2-h_feats --dir_b WSI/IgA/labels_classification
    python sync_dirs.py --dir_a WSI/IgA/UNI2-h_feats --dir_b WSI/IgA/labels_classification --dry_run
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Delete files in dir_b whose stem is absent from dir_a."
    )
    parser.add_argument("--dir_a", required=True, help="Reference directory.")
    parser.add_argument("--dir_b", required=True, help="Directory to clean up.")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print what would be deleted without deleting."
    )
    args = parser.parse_args()

    stems_a = {os.path.splitext(f)[0] for f in os.listdir(args.dir_a)}
    files_b = [f for f in os.listdir(args.dir_b) if os.path.isfile(os.path.join(args.dir_b, f))]

    to_delete = [f for f in files_b if os.path.splitext(f)[0] not in stems_a]

    if not to_delete:
        print("Nothing to delete.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to delete from {args.dir_b}: {len(to_delete)}")
    for f in sorted(to_delete):
        path = os.path.join(args.dir_b, f)
        print(f"  {'(would delete)' if args.dry_run else 'deleting'} {f}")
        if not args.dry_run:
            os.remove(path)


if __name__ == "__main__":
    main()
