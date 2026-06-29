"""Rename entries in a name list using an old_name -> new_name mapping CSV.

Usage:
    python rename_from_mapping.py names.csv mapping.csv [-o output.csv]

names.csv   : no header, one name per row (first column used).
mapping.csv : header row with columns `old_name` and `new_name`.

Names found in `old_name` are replaced by the corresponding `new_name`;
names not in the mapping are left unchanged. Result is written to output
(default: overwrites names.csv).
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", help="CSV with no header, list of names to process")
    parser.add_argument("mapping", help="CSV with header columns old_name,new_name")
    parser.add_argument("-o", "--output", help="output CSV (default: overwrite names file)")
    args = parser.parse_args()

    with open(args.mapping, newline="", encoding="utf-8-sig") as f:
        mapping = {row["old_name"]: row["new_name"] for row in csv.DictReader(f)}

    with open(args.names, newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]

    renamed = 0
    for row in rows:
        if row[0] in mapping:
            row[0] = mapping[row[0]]
            renamed += 1

    out = args.output or args.names
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Renamed {renamed} of {len(rows)} entries -> {out}")


if __name__ == "__main__":
    main()
