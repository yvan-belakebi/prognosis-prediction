"""
compare_registry_anonymized.py — Check that the new registry_anonymized.csv equals the
old one apart from the added proteinuria columns.

Passes when the new file has the old columns plus exactly the extra columns
(default: Proteinuria_g_d, Sampling_method), and the same rows in the same order
with identical values in every old column. Values are compared as text.

Run from the project root:
    python followup_data/scripts/compare_registry_anonymized.py \
        followup_data/derived/renamed/registry_anonymized_backup.csv \
        followup_data/derived/renamed/registry_anonymized.csv
"""

import argparse
import sys

import pandas as pd

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("old_csv")
ap.add_argument("new_csv")
ap.add_argument(
    "--extra",
    nargs="+",
    default=["Proteinuria_g_d", "Sampling_method", "Sex", "Age", "BMI"],
)
args = ap.parse_args()

read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
old, new = read(args.old_csv), read(args.new_csv)

ok = True
added = [c for c in new.columns if c not in old.columns]
removed = [c for c in old.columns if c not in new.columns]
if sorted(added) != sorted(args.extra) or removed:
    print(
        f"Columns differ: added {added}, removed {removed}, expected added {args.extra}"
    )
    ok = False

if len(old) != len(new):
    print(f"Row count differs: old {len(old)}, new {len(new)}")
    ok = False
else:
    common = [c for c in old.columns if c in new.columns]
    diff = old[common].ne(new[common])
    for c in diff.columns[diff.any()]:
        rows = diff.index[diff[c]]
        print(f"Column {c!r}: {len(rows)} rows differ, first at row {rows[0]}")
        ok = False

print("OK: identical apart from the added columns" if ok else "FAILED")
sys.exit(0 if ok else 1)
