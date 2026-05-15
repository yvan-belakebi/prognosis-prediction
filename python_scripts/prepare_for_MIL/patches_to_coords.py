import argparse
import os
import re
import numpy as np

# ran with arguments --root_dir WSI_full/IgA

PATCH_FILENAME_RE = re.compile(
    r"^(?P<row>\d+)[-_](?P<col>\d+)\.(?:jpe?g|png)$", re.IGNORECASE
)


def parse_patch_name(filename: str):
    match = PATCH_FILENAME_RE.match(filename)
    if not match:
        return None
    row = int(match.group("row"))
    col = int(match.group("col"))
    return row, col


def collect_patch_coords(folder_path: str, tile_size: int):
    entries = sorted(os.listdir(folder_path))
    coords = []
    for entry in entries:
        full_path = os.path.join(folder_path, entry)
        if not os.path.isfile(full_path):
            continue
        parsed = parse_patch_name(entry)
        if parsed is None:
            continue
        row, col = parsed
        coords.append((row * tile_size, col * tile_size))

    if not coords:
        return None

    coords_array = np.array(coords, dtype=np.int32)
    return coords_array


def save_folder_coords(folder_path: str, tile_size: int, output_dir: str):
    coords = collect_patch_coords(folder_path, tile_size)
    if coords is None:
        return False
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(folder_path)}.npy")
    np.save(output_path, coords)
    return True


def find_patch_folders(root_folder: str):
    for entry in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, entry)
        if os.path.isdir(folder_path):
            yield folder_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a .npy coordinate array for each patch folder."
    )
    parser.add_argument(
        "root_folder",
        help="Root folder containing patch subfolders such as wsi_1, wsi_2, ...",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Tile size used to compute coordinates (default: 224).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden directories when searching for patch folders.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_folder):
        raise SystemExit(f"Root folder does not exist: {args.root_folder}")

    patches_root = os.path.join(args.root_folder, "single")
    if not os.path.isdir(patches_root):
        raise SystemExit(f"Patches folder does not exist: {patches_root}")

    processed = 0
    skipped = 0
    coords_folder = os.path.join(args.root_folder, "coords")
    for folder_path in find_patch_folders(patches_root):
        if not args.include_hidden and os.path.basename(folder_path).startswith("."):
            continue
        saved = save_folder_coords(folder_path, args.tile_size, coords_folder)
        if saved:
            print(
                f"Saved coordinates: {os.path.join(coords_folder, os.path.basename(folder_path) + '.npy')}"
            )
            processed += 1
        else:
            print(f"Skipped folder (no patch files found): {folder_path}")
            skipped += 1

    print(f"Done. Processed {processed} patch folders, skipped {skipped} folders.")


if __name__ == "__main__":
    main()
