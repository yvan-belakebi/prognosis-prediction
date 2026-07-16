"""Write a slide-list CSV of the IgA slides in the anonymized registry.

Reads renamed/registry_anonymized.csv, keeps the rows where is_IgA is true, and
writes their ANON_name out as a `wsi_anon_name` column -- the one input
tiling_from_csv_folders.py reads to look each slide up in the registry.
"""

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.path.join(
    HERE, "..", "followup_data", "derived", "renamed", "registry_anonymized.csv"
)


def iga_slide_names(registry_csv):
    """Return the deduplicated ANON_names of the registry's IgA slides, in order."""
    seen = {}
    with open(registry_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("is_IgA") or "").strip().lower() != "true":
                continue
            name = (row.get("ANON_name") or "").strip()
            if name:
                seen[name] = None
    return list(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", default=DEFAULT_REGISTRY, help="anonymized registry CSV"
    )
    parser.add_argument(
        "-o",
        "--out",
        default=os.path.join(HERE, "iga_slides_from_registry.csv"),
        help="output slide-list CSV",
    )
    args = parser.parse_args()

    names = iga_slide_names(args.registry)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_anon_name"])
        writer.writerows([name] for name in names)
    print(f"{len(names)} IgA slides -> {args.out}")


if __name__ == "__main__":
    main()
