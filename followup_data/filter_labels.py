import pandas as pd
import sys
from pathlib import Path

def filter_csv(input_path, output_path, registry_dir):
    df = pd.read_csv(input_path)
    registry = Path(registry_dir)

    # Build the set of existing "biopsy/slide" paths once, so we don't
    # hit the filesystem for every row.
    existing = set()
    for biopsy_dir in registry.iterdir():
        if biopsy_dir.is_dir():
            for slide in biopsy_dir.iterdir():
                existing.add(f"{biopsy_dir.name}/{slide.name}")
                # Also index by stem so CSV entries without extensions still match
                existing.add(f"{biopsy_dir.name}/{slide.stem}")

    mask = df["file_name"].astype(str).isin(existing)
    filtered = df[mask]

    filtered.to_csv(output_path, index=False)
    print(f"Kept {len(filtered)} of {len(df)} rows -> {output_path}")

    missing = df.loc[~mask, "file_name"].tolist()
    if missing:
        print(f"Dropped {len(missing)} rows not found in {registry_dir}.")
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filter.py input.csv output.csv IgA_registry")
        sys.exit(1)
    filter_csv(sys.argv[1], sys.argv[2], sys.argv[3])