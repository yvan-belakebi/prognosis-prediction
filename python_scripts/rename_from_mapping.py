"""Rename entries in a name list using an old_name -> new_name mapping CSV.

Usage:
    python rename_from_mapping.py names.csv mapping.csv [-o output.csv]

names.csv   : no header, one name per row (first column used).
mapping.csv : header row with columns `old_name` and `new_name`.

Matching is done on the basename without extension, so the extension may
differ or be absent on either side. The original name's extension (if any)
is preserved on the renamed result. Names not in the mapping are left
unchanged. Result is written to output (default: overwrites names.csv).
"""

import argparse
import csv
import re

# Strip a trailing extension only when it looks like a real one (contains a
# letter), so numeric timestamps like "...21.31.46" are left intact even when
# the other side carries no extension.
_EXT_RE = re.compile(r"\.(?=[^.]*[A-Za-z])[A-Za-z0-9]{1,8}$")


def stem(name):
    return _EXT_RE.sub("", name)


def ext(name):
    s = stem(name)
    return name[len(s):]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", help="CSV with no header, list of names to process")
    parser.add_argument("mapping", help="CSV with header columns old_name,new_name")
    parser.add_argument("-o", "--output", help="output CSV (default: overwrite names file)")
    args = parser.parse_args()

    with open(args.mapping, newline="", encoding="utf-8-sig") as f:
        mapping = {stem(row["old_name"]): row["new_name"] for row in csv.DictReader(f)}

    with open(args.names, newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]

    renamed = 0
    for row in rows:
        new = mapping.get(stem(row[0]))
        if new is not None:
            row[0] = stem(new) + ext(row[0])
            renamed += 1

    out = args.output or args.names
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Renamed {renamed} of {len(rows)} entries -> {out}")


if __name__ == "__main__":
    main()
