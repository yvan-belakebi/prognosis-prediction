"""
sync_dirs.py — Remove files/folders from dir B that have no matching name in dir A.

For files, only the stem (name without extension) is compared, so A/foo.h5
keeps B/foo.npy. For folders, the folder name is compared as-is, and matches
against either folder names or file stems in A (so A/foo.h5 also keeps B/foo/).
Use --dry_run to preview deletions before committing.

Usage:
    python sync_dirs.py --dir_a WSI/IgA/UNI2-h_feats --dir_b WSI/IgA/labels_classification
    python sync_dirs.py --dir_a WSI/IgA/UNI2-h_feats --dir_b WSI/IgA/labels_classification --dry_run
"""

import argparse
import os
import shutil


def entry_name(path, name):
    """Comparison name: stem for files, folder name as-is for directories."""
    return name if os.path.isdir(path) else os.path.splitext(name)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Delete files/folders in dir_b whose name is absent from dir_a."
    )
    parser.add_argument("--dir_a", required=True, help="Reference directory.")
    parser.add_argument("--dir_b", required=True, help="Directory to clean up.")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print what would be deleted without deleting."
    )
    args = parser.parse_args()

    names_a = {
        entry_name(os.path.join(args.dir_a, e), e) for e in os.listdir(args.dir_a)
    }

    to_delete = []
    for e in os.listdir(args.dir_b):
        path = os.path.join(args.dir_b, e)
        if entry_name(path, e) not in names_a:
            to_delete.append((e, path, os.path.isdir(path)))

    if not to_delete:
        print("Nothing to delete.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Entries to delete from {args.dir_b}: {len(to_delete)}")
    for e, path, is_dir in sorted(to_delete):
        kind = "folder" if is_dir else "file"
        print(f"  {'(would delete)' if args.dry_run else 'deleting'} {kind} {e}")
        if not args.dry_run:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)


if __name__ == "__main__":
    main()