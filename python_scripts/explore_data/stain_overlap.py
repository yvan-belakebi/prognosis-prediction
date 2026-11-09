"""
stain_overlap.py — How often a biopsy carries redundant stains, from the labels CSV.

Reports two overlaps, per biopsy (one row per slide in the CSV):
  * HE and HES: biopsies with at least one HE slide and at least one HES slide;
  * trichromes: biopsies with two or more distinct trichrome stains (e.g. Masson
    trichrome + AFOG), and biopsies with two or more trichrome slides of any kind.

Each proportion is given over all biopsies and over the biopsies that have at
least one of the stains involved. For each group (HE/HES, trichromes), the
biopsies with two or more slides of the group are also split into those whose
slides are all one stain, repeated, and those with slides of distinct stains. Stain names are compared case- and
punctuation-insensitively ("Masson Trichrom" = "masson-trichrom").

Usage:
    python python_scripts/explore_data/stain_overlap.py --labels_csv label_csvs/labels.csv
    # older label CSVs name the column "Stain":
    python python_scripts/explore_data/stain_overlap.py \
        --labels_csv label_csvs/labels_regression.csv --stain_col Stain
"""

import argparse
import re

import pandas as pd

# Trichrome-type connective-tissue stains found in the registry's stain column.
TRICHROMES = ["Masson Trichrom", "Masson Goldner", "AFOG", "MSB", "Lendrum"]


def canon_stain(name):
    """Same key as mil_utils.canon_stain (not imported: it pulls in torch)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def share(k, n):
    return f"{k:6d} / {n:6d} = {100 * k / n:5.1f}%" if n else f"{k:6d} / {n:6d}"


def multi_slide_breakdown(stains, group, display):
    """Print, for biopsies with >= 2 slides of `group`, one stain repeated vs distinct stains."""
    in_group = stains.agg(lambda k: k[k.isin(group)].tolist())
    multi = in_group[in_group.map(len) >= 2]
    kinds = multi.map(lambda k: len(set(k)))
    print(f"    biopsies with >= 2 slides of these stains:  {len(multi)}")
    print(f"      all one stain, repeated:                  {share((kinds == 1).sum(), len(multi))}")
    print(f"      slides from distinct stains:              {share((kinds >= 2).sum(), len(multi))}")
    repeated = multi[kinds == 1].map(lambda k: display.get(k[0], k[0]))
    for stain, k in repeated.value_counts().items():
        print(f"        {stain} only: {k}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_csv", default="label_csvs/labels.csv")
    p.add_argument("--biopsy_col", default="biopsy_number")
    p.add_argument("--stain_col", default="stain")
    p.add_argument("--trichromes", nargs="+", default=TRICHROMES,
                   help="Stain names counted as trichromes.")
    p.add_argument("--by_split", action="store_true",
                   help="Also report each value of the `split` column separately.")
    args = p.parse_args()

    table = pd.read_csv(args.labels_csv, dtype=str, usecols=lambda c: c in
                        {args.biopsy_col, args.stain_col, "split"})
    table = table.dropna(subset=[args.biopsy_col, args.stain_col])
    table["key"] = table[args.stain_col].map(canon_stain)

    groups = [("all", table)]
    if args.by_split and "split" in table:
        groups += [(f"split = {s}", g) for s, g in table.groupby("split")]

    trich = {canon_stain(s) for s in args.trichromes}
    display = dict(zip(map(canon_stain, args.trichromes), args.trichromes))
    for name, t in groups:
        stains = t.groupby(args.biopsy_col)["key"]
        keys = stains.agg(set)
        n = len(keys)
        has_he = keys.map(lambda s: "he" in s)
        has_hes = keys.map(lambda s: "hes" in s)
        n_trich_kinds = keys.map(lambda s: len(s & trich))
        n_trich_slides = stains.agg(lambda k: k.isin(trich).sum())

        print(f"\n[{name}] {len(t)} slides, {n} biopsies ({args.labels_csv})")
        print("  HE and HES")
        print(f"    over all biopsies:            {share((has_he & has_hes).sum(), n)}")
        print(f"    over biopsies with HE or HES: "
              f"{share((has_he & has_hes).sum(), (has_he | has_hes).sum())}")
        print(f"    (HE only {(has_he & ~has_hes).sum()}, HES only {(~has_he & has_hes).sum()})")
        multi_slide_breakdown(stains, {"he", "hes"}, {"he": "HE", "hes": "HES"})

        any_trich = n_trich_slides > 0
        print(f"  Trichromes ({', '.join(args.trichromes)})")
        print(f"    >= 2 distinct trichrome stains, over all biopsies:        "
              f"{share((n_trich_kinds >= 2).sum(), n)}")
        print(f"    >= 2 distinct trichrome stains, over biopsies with one:   "
              f"{share((n_trich_kinds >= 2).sum(), any_trich.sum())}")
        print(f"    >= 2 trichrome slides (any kind), over biopsies with one: "
              f"{share((n_trich_slides >= 2).sum(), any_trich.sum())}")
        multi_slide_breakdown(stains, trich, display)
        combos = keys[n_trich_kinds >= 2].map(lambda s: " + ".join(sorted(display[k] for k in s & trich)))
        if not combos.empty:
            print("    most common combinations:")
            for combo, k in combos.value_counts().head(5).items():
                print(f"      {combo:35s} {k}")


if __name__ == "__main__":
    main()
