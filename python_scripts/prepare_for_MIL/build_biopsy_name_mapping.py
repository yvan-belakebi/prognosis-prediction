"""
build_biopsy_name_mapping.py — Recover the unambiguous biopsy directory name for every
registry slide, undoing the identity loss in define_labels.transform_label.

Why this exists
---------------
``transform_label`` matches the leading letters of a biopsy number but never captures
them::

    re.match(r"^[A-Za-z]{1,2}(\\d{2})(\\d+)$", label)   # group 1 = year, group 2 = number

Those letters are a laboratory code, and each lab numbers its biopsies independently
within a year, so dropping them makes distinct biopsies collide::

    B1310959   -> "10959/13" -> 10959-13      patient A
    BG1310959  -> "10959/13" -> 10959-13      patient B

``reorganize_wsi_dirs.rename_dirs_in`` then treats the resulting name clash as two
source folders for one biopsy and *merges* them (tagging the line ``[MERGE]``), so two
patients' slides end up in one directory.  ``index_biopsies`` groups by that directory,
so one bag mixes two patients and takes its label from whichever slide is found first.

The fix is mechanical because nothing was actually lost at slide level: in
``registry_anonymized.csv`` every ``ANON_name`` is unique and maps to exactly one raw
``biop_number``.  This script writes that slide -> raw-biopsy-number mapping so the
directories can be split without any manual disambiguation.

Scope: the registry cohorts only (WSI/IgA_registry, WSI/non_IgA).  The IgA cohort needs
no fix — its one normalisation clash ('1974-14' from 'B14 1974' + 'B1401974') is a single
biopsy whose number is written two ways, re-cut into a second specimen block years later,
so merging those slides is correct.

Output CSV (one row per slide):

    source        registry | non_IgA
    slide_name    ANON_name — the on-disk file stem
    current_dir   biopsy directory the slide sits in now (transform_label output)
    target_dir    raw biop_number — the unambiguous directory name
    patient       ID_diagnosis, carried through so the split can be verified
    collision     True when current_dir holds more than one target_dir

Run from the project root:
    python python_scripts/prepare_for_MIL/build_biopsy_name_mapping.py

This script only reads the registry CSV and writes the mapping — it renames nothing.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from define_labels import transform_label, biopsy_to_dirname  # noqa: E402


def build_mapping(registry_csv):
    """Return the slide-level mapping DataFrame for both registry cohorts."""
    df = pd.read_csv(registry_csv, low_memory=False)
    for col in ("biop_number", "ANON_name", "is_IgA", "ID_diagnosis"):
        if col not in df.columns:
            raise ValueError(f"{registry_csv} must contain a '{col}' column.")

    raw = df["biop_number"].astype(str).str.strip()
    # The raw numbers are plain alphanumerics (verified: no whitespace or separators),
    # so they are already usable verbatim as directory names.
    unsafe = raw[raw.str.contains(r"[^A-Za-z0-9]", regex=True)]
    if len(unsafe):
        raise ValueError(
            f"{len(unsafe)} biop_number values are not filesystem-safe, e.g. "
            f"{unsafe.unique()[:5].tolist()} - extend this script before renaming."
        )

    out = pd.DataFrame(
        {
            "source": df["is_IgA"].map({True: "registry", False: "non_IgA"}),
            "slide_name": df["ANON_name"].astype(str),
            "current_dir": raw.apply(transform_label).apply(biopsy_to_dirname),
            "target_dir": raw,
            "patient": df["ID_diagnosis"],
        }
    )

    # A collision is a current_dir that resolves to more than one raw biopsy number.
    n_targets = out.groupby("current_dir")["target_dir"].transform("nunique")
    out["collision"] = n_targets > 1
    return out


def verify(m):
    """Check the mapping is a usable rename plan; print a summary. Returns True if sane."""
    ok = True
    dup = m.slide_name.duplicated().sum()
    print(f"slides: {len(m)} | unique slide_name: {m.slide_name.nunique()}")
    if dup:
        print(f"  [FAIL] {dup} duplicated slide_name rows - a slide cannot have two targets")
        ok = False

    print(f"current dirs: {m.current_dir.nunique()} -> target dirs: {m.target_dir.nunique()}")
    col = m[m.collision]
    print(
        f"collided current dirs: {col.current_dir.nunique()} "
        f"({len(col)} slides, {col.patient.nunique()} patients) -> "
        f"{col.target_dir.nunique()} target dirs"
    )

    # The point of the whole exercise: one patient per target directory.
    per_target = m.groupby("target_dir")["patient"].nunique()
    bad = per_target[per_target > 1]
    if len(bad):
        print(f"  [FAIL] {len(bad)} target dirs still hold >1 patient: {bad.index[:5].tolist()}")
        ok = False
    else:
        print("  [OK]  every target dir holds exactly one patient")

    # No target may collide with an unrelated current dir, or the rename could
    # move slides into a directory that is itself still waiting to be renamed.
    clash = set(m.target_dir) & set(m.current_dir)
    if clash:
        print(f"  [WARN] {len(clash)} target names equal an existing current dir: {sorted(clash)[:5]}")
    else:
        print("  [OK]  no target name collides with an existing current dir")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Build the slide -> unambiguous biopsy directory mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="followup_data/derived/renamed/biopsy_name_mapping.csv",
    )
    parser.add_argument(
        "--collisions_csv",
        default="followup_data/derived/renamed/biopsy_name_collisions.csv",
        help="Separate listing of just the collided directories, for review.",
    )
    args = parser.parse_args()

    m = build_mapping(args.registry_csv)
    print()
    ok = verify(m)
    print()

    for path, frame in (
        (args.output_csv, m),
        (args.collisions_csv, m[m.collision].sort_values(["current_dir", "target_dir"])),
    ):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"wrote {path}  ({len(frame)} rows)")

    print()
    print("--- collided directories ---")
    col = m[m.collision]
    for cur, g in col.groupby("current_dir"):
        parts = [
            f"{t} ({len(gg)} slides, patient {gg.patient.iloc[0]:.0f})"
            for t, gg in g.groupby("target_dir")
        ]
        print(f"  {cur:>12}  ->  " + "  |  ".join(parts))

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
