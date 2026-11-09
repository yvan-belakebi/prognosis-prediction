"""
stain_overlap.py — How often a biopsy carries redundant stains, from the labels CSV.

Reports two overlaps, per biopsy (one row per slide in the CSV):
  * HE and HES, and PAS and PASM: biopsies with at least one slide of each;
  * trichromes: biopsies with two or more distinct trichrome stains (e.g. Masson
    trichrome + AFOG), and biopsies with two or more trichrome slides of any kind.

Each proportion is given over all biopsies and over the biopsies that have at
least one of the stains involved. For each group (HE/HES, PAS/PASM, trichromes), the
biopsies with two or more slides of the group are also split into those whose
slides are all one stain, repeated, and those with slides of distinct stains. Stain names are compared case- and
punctuation-insensitively ("Masson Trichrom" = "masson-trichrom").

Complete panel: the largest set of biopsies that have at least one slide of
every stain of --panel, i.e. the biopsies whose stains cover the whole panel.
A panel entry merges stains with '|' (first name = display name), as in
stain_config.sh; by default HE and HES form one stain, and so do the
--trichromes. Then stains are dropped one at a time, each time the one whose
removal adds the most complete biopsies, to show which stains limit the panel.

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
# stain_config.sh's stains; "Trichrome" is expanded to --trichromes.
PANEL = ["HE_HES|HE|HES", "PAS", "PASM", "Congo", "Trichrome", "Sirius red",
         "C1q", "C3", "C5-9", "IgA", "IgG", "IgM", "Kappa", "Lambda"]


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


def pair_overlap(stains, keys, a, b):
    """Print how often a biopsy has both stain a and stain b, and the repeat breakdown."""
    ka, kb = canon_stain(a), canon_stain(b)
    has_a, has_b = keys.map(lambda s: ka in s), keys.map(lambda s: kb in s)
    both = int((has_a & has_b).sum())
    print(f"  {a} and {b}")
    for label, base in [("over all biopsies", len(keys)),
                        (f"over biopsies with {a} or {b}", int((has_a | has_b).sum()))]:
        print(f"    {label + ':':32s}{share(both, base)}")
    print(f"    ({a} only {(has_a & ~has_b).sum()}, {b} only {(~has_a & has_b).sum()})")
    multi_slide_breakdown(stains, {ka, kb}, {ka: a, kb: b})


def complete_panel(keys, panel, events=None, event_types=None):
    """Print how many biopsies cover every stain of panel, then drop stains greedily.

    keys: per-biopsy set of canonical stain keys; panel: {display name: set of keys};
    events / event_types: optional per-biopsy event flag (0/1) and event type,
    indexed like keys, to count the events of the complete biopsies.
    """
    has = pd.DataFrame({name: keys.map(lambda s, g=group: bool(s & g))
                        for name, group in panel.items()})
    n = len(has)

    def n_events(mask):
        return "" if events is None else f"  events {int(events[mask].sum()):5d}"

    complete = has.all(axis=1)
    print(f"  Complete panel ({len(panel)} stains: {', '.join(panel)})")
    print(f"    biopsies with every stain:  {share(int(complete.sum()), n)}{n_events(complete)}")
    if events is not None:
        k, m = int(events[complete].sum()), int(complete.sum())
        print(f"    events among them:          {share(k, m)}  "
              f"(all biopsies: {100 * events.mean():.1f}%)")
        if event_types is not None:
            types = event_types[complete & (events == 1)].value_counts()
            print("      by type: " + ", ".join(f"{t} {c}" for t, c in types.items()))
    print("    stains present, per biopsy: "
          + ", ".join(f"{name} {100 * has[name].mean():.0f}%" for name in panel))
    print("    dropping the most limiting stain each time:")
    left = list(panel)
    while len(left) > 1:
        gain = {c: int(has[[x for x in left if x != c]].all(axis=1).sum()) for c in left}
        drop = max(gain, key=gain.get)
        left.remove(drop)
        print(f"      without {drop:12s} -> {len(left):2d} stains: {share(gain[drop], n)}"
              f"{n_events(has[left].all(axis=1))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_csv", default="label_csvs/labels.csv")
    p.add_argument("--biopsy_col", default="biopsy_number")
    p.add_argument("--stain_col", default="stain")
    p.add_argument("--trichromes", nargs="+", default=TRICHROMES,
                   help="Stain names counted as trichromes.")
    p.add_argument("--panel", nargs="+", default=None,
                   help="Stains of the complete panel, '|' merging names into one "
                   "(default: stain_config.sh's, with HE|HES and the --trichromes merged).")
    p.add_argument("--event_col", default="event",
                   help="Event flag column (1 = event), for the event counts of the "
                   "complete panel; skipped when the CSV has none (regression labels).")
    p.add_argument("--by_split", action="store_true",
                   help="Also report each value of the `split` column separately.")
    args = p.parse_args()

    table = pd.read_csv(args.labels_csv, dtype=str, usecols=lambda c: c in
                        {args.biopsy_col, args.stain_col, "split", args.event_col, "event_type"})
    if args.event_col not in table:
        print(f"(no '{args.event_col}' column in {args.labels_csv}: event counts skipped)")
    table = table.dropna(subset=[args.biopsy_col, args.stain_col])
    table["key"] = table[args.stain_col].map(canon_stain)

    groups = [("all", table)]
    if args.by_split and "split" in table:
        groups += [(f"split = {s}", g) for s, g in table.groupby("split")]

    trich = {canon_stain(s) for s in args.trichromes}
    panel = {}
    for spec in args.panel or PANEL:
        names = (["Trichrome"] + args.trichromes if spec == "Trichrome" and not args.panel
                 else [x.strip() for x in spec.split("|")])
        panel[names[0]] = {canon_stain(x) for x in names}
    display = dict(zip(map(canon_stain, args.trichromes), args.trichromes))
    for name, t in groups:
        stains = t.groupby(args.biopsy_col)["key"]
        keys = stains.agg(set)
        n = len(keys)
        n_trich_kinds = keys.map(lambda s: len(s & trich))
        n_trich_slides = stains.agg(lambda k: k.isin(trich).sum())

        print(f"\n[{name}] {len(t)} slides, {n} biopsies ({args.labels_csv})")
        for a, b in [("HE", "HES"), ("PAS", "PASM")]:
            pair_overlap(stains, keys, a, b)

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

        events = event_types = None
        if args.event_col in t:
            # An event belongs to the biopsy: every slide carries the same flag.
            by_biopsy = t.groupby(args.biopsy_col)
            events = by_biopsy[args.event_col].agg(
                lambda e: int(pd.to_numeric(e, errors="coerce").max() > 0))
            if "event_type" in t:
                event_types = by_biopsy["event_type"].first().fillna("")
        complete_panel(keys, panel, events, event_types)


if __name__ == "__main__":
    main()
