"""
reorganize_wsi_dirs.py — Restructure flat WSI slide directories to biopsy-nested layout.

Before: WSI/IgA/UNI2-h_feats/slide_name.h5
After:  WSI/IgA/UNI2-h_feats/<biopsy_dir>/slide_name.h5

Biopsy directories are derived from:
  - IgA cohort     : followup_data/derived/renamed/IgA_full_data.csv   (Biopsy Number + File Location)
  - Registry cohort: followup_data/derived/renamed/registry_anonymized.csv  (biop_number + ANON_name)

By default runs in dry-run mode — prints planned moves without touching files.
Pass --apply to perform the actual moves.

Run from the project root:
    # Dry run (default) — inspect what would happen
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py \\
        --iga_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
        --registry_dirs WSI/IgA_registry/UNI2-h_feats WSI/IgA_registry/labels \\
                        WSI/non_IgA/UNI2-h_feats WSI/non_IgA/labels

    # Apply changes
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py \\
        --iga_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
        --registry_dirs WSI/IgA_registry/UNI2-h_feats WSI/IgA_registry/labels \\
                        WSI/non_IgA/UNI2-h_feats WSI/non_IgA/labels \\
        --apply


Rename mode (--rename)
----------------------
Some slides are named after scanner timestamps, e.g. "2024-06-13 11.40.31.h5".
The embedded dots are harmless (os.path.splitext / discover_bags handle them),
but the *space* is fragile: it breaks shell helpers (copy_files.sh, unquoted
paths) and is an easy source of silent CSV-match failures where a slide is
quietly dropped from training.

--rename assigns each *unsafe* slide a filesystem-safe, biopsy-based sequential
name (e.g. '12-23_s01', or 'unknown_s0001' when the biopsy is not known) and:
  - writes a mapping CSV  (biopsy_number, old_name, new_name),
  - renames the matching files across every --slide_dirs entry (feature, label,
    coord and raw-WSI dirs), in both flat and biopsy-nested layouts, using one
    consistent global plan,
  - writes rewritten *copies* of the source metadata CSVs (IgA File Location /
    registry ANON_name updated to the new names) into --csv_out_dir, which
    downstream tasks then consume instead of the originals.

Already-safe names (e.g. '32980023') are left untouched unless --rename-all is
passed. Like the move mode, --rename is a dry run until you add --apply.

    # Dry run — see which slides would be renamed
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename \\
        --slide_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
                     data/raw_wsi/IgA \\
        --slide_exts .npy .h5 .svs

    # Apply
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename \\
        --slide_dirs WSI/IgA/UNI2-h_feats WSI/IgA/labels WSI/IgA/coords \\
                     data/raw_wsi/IgA \\
        --slide_exts .npy .h5 .svs \\
        --apply


Directory-rename mode (--rename_dirs)
-------------------------------------
Normalises the *biopsy-level subdirectory* names (not the files inside them) of
each --slide_dirs entry, applying transform_label() then biopsy_to_dirname() to
every first-level subdirectory (e.g. "B2312" → "12-23"). When the normalised
name already exists as a sibling directory, the two are merged (items moved
across, source removed). Like the other modes, it is a dry run until --apply.

    # Dry run — see which biopsy dirs would be renamed
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename_dirs \\
        --slide_dirs WSI/IgA/labels WSI/IgA/coords WSI/IgA/UNI2-h_feats

    # Apply
    python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename_dirs \\
        --slide_dirs WSI/IgA/labels WSI/IgA/coords WSI/IgA/UNI2-h_feats \\
        --apply
"""

import argparse
import os
import re
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from define_labels import (
    biopsy_to_dirname,
    extract_file_name,
    transform_label,
)  # noqa: E402

SLIDE_EXTS = {".npy", ".h5"}


# ---------------------------------------------------------------------------
# Build slide-stem → biopsy-dir mappings from CSV metadata
# ---------------------------------------------------------------------------


def build_iga_mapping(iga_slides_csv: str) -> dict:
    """Return {slide_stem: biopsy_dir} for the IgA cohort."""
    df = pd.read_csv(iga_slides_csv)
    mapping = {}
    for _, row in df.iterrows():
        stem = extract_file_name(row["File Location"])
        biopsy_dir = biopsy_to_dirname(str(row["Biopsy Number"]))  # transform_label()
        if stem and not pd.isna(stem):
            mapping[stem] = biopsy_dir
    return mapping


def build_registry_mapping(registry_csv: str) -> dict:
    """Return {slide_stem: biopsy_dir} for the full registry (IgA + non-IgA)."""
    df = pd.read_csv(registry_csv, usecols=["ANON_name", "biop_number"])
    mapping = {}
    for _, row in df.iterrows():
        stem = str(row["ANON_name"]).strip()
        biopsy_dir = biopsy_to_dirname(str(row["biop_number"]))  # transform_label()
        if stem:
            mapping[stem] = biopsy_dir
    return mapping


# ---------------------------------------------------------------------------
# Rename mode: assign filesystem-safe slide names + rewrite metadata CSVs
# ---------------------------------------------------------------------------

# A stem is "unsafe" if it has surrounding whitespace or any character outside
# [word-char, dot, hyphen]. Internal dots (e.g. '11.40.31') are kept — only the
# *space* and other punctuation actually break shells / CSV matching.
_UNSAFE_STEM_RE = re.compile(r"[^\w.\-]")


def is_unsafe_stem(stem: str) -> bool:
    """True if `stem` needs renaming to be safe in shells and CSV joins."""
    return stem != stem.strip() or bool(_UNSAFE_STEM_RE.search(stem))


def build_iga_records(iga_slides_csv: str) -> list:
    """Return [(slide_stem, biopsy_number)] for the IgA cohort.

    biopsy_number is the normalised 'number/year' form (transform_label), or ""
    when missing. Mirrors the stem/biopsy derivation in define_labels.py.
    """
    df = pd.read_csv(iga_slides_csv)
    records = []
    for _, row in df.iterrows():
        stem = extract_file_name(row["File Location"])
        if not stem or pd.isna(stem):
            continue
        biopsy = transform_label(str(row["Biopsy Number"]))
        records.append((str(stem), "" if pd.isna(biopsy) else str(biopsy)))
    return records


def build_registry_records(registry_csv: str) -> list:
    """Return [(slide_stem, biopsy_number)] for the full registry cohort."""
    df = pd.read_csv(registry_csv, usecols=["ANON_name", "biop_number"])
    records = []
    for _, row in df.iterrows():
        stem = str(row["ANON_name"]).strip()
        if not stem or stem.lower() == "nan":
            continue
        biopsy = transform_label(str(row["biop_number"]))
        records.append((stem, "" if pd.isna(biopsy) else str(biopsy)))
    return records


def discover_stems(directory: str, exts: set) -> set:
    """Return the set of slide stems found directly under `directory` or one
    level of subdirectories (flat + biopsy-nested), filtered by extension."""
    stems = set()
    if not os.path.isdir(directory):
        return stems
    for entry in os.scandir(directory):
        if entry.is_file():
            stem, ext = os.path.splitext(entry.name)
            if ext in exts:
                stems.add(stem)
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file():
                    stem, ext = os.path.splitext(sub.name)
                    if ext in exts:
                        stems.add(stem)
    return stems


def assign_new_names(records: dict, existing_names: set, rename_all: bool) -> dict:
    """Return {old_stem: new_stem} for every stem that needs a safe name.

    records       : {old_stem: biopsy_number}
    existing_names: every stem already in use (so new names never collide).
    rename_all    : if False, only unsafe stems are renamed.

    New names are biopsy-scoped sequential ids: '<biopsy_dir>_s<NN>'
    (e.g. '12-23_s01'), or 'unknown_s<NNNN>' when the biopsy is unknown.
    """
    used = set(existing_names)
    per_biopsy_counter = {}
    plan = {}
    for old_stem in sorted(records):
        if not rename_all and not is_unsafe_stem(old_stem):
            continue
        biopsy = records[old_stem]
        bdir = biopsy_to_dirname(biopsy) if biopsy else "unknown"
        n = per_biopsy_counter.get(bdir, 0) + 1
        while True:
            candidate = f"{bdir}_s{n:02d}" if bdir != "unknown" else f"unknown_s{n:04d}"
            if candidate not in used:
                break
            n += 1
        per_biopsy_counter[bdir] = n
        used.add(candidate)
        plan[old_stem] = candidate
    return plan


def write_mapping_csv(plan: dict, records: dict, mapping_out: str, apply: bool):
    """Write the biopsy_number / old_name / new_name mapping CSV."""
    rows = [
        {
            "biopsy_number": records.get(old_stem, ""),
            "old_name": old_stem,
            "new_name": new_stem,
        }
        for old_stem, new_stem in plan.items()
    ]
    df = pd.DataFrame(rows).sort_values("new_name").reset_index(drop=True)
    if apply:
        os.makedirs(os.path.dirname(mapping_out) or ".", exist_ok=True)
        df.to_csv(mapping_out, index=False)
        print(f"[MAP  ] Wrote {len(df)} mappings → {mapping_out}\n")
    else:
        print(f"[DRY  ] Would write {len(df)} mappings → {mapping_out}")
        print(df.head(10).to_string(index=False))
        if len(df) > 10:
            print(f"  ... ({len(df) - 10} more)")
        print()
    return df


def _rename_file(file_path: str, name: str, parent: str, plan: dict, exts: set,
                 apply: bool) -> int:
    """Rename a single slide file in place (same dir) per `plan`. Returns 1 if
    the file was (or would be) renamed, else 0."""
    stem, ext = os.path.splitext(name)
    if ext not in exts:
        return 0
    new_stem = plan.get(stem)
    if new_stem is None:
        return 0
    dest = os.path.join(parent, new_stem + ext)
    if apply:
        if os.path.exists(dest):
            print(f"  [WARN ] Destination exists, skipping: {dest}")
            return 0
        shutil.move(file_path, dest)
        print(f"  [REN  ] {name}  →  {new_stem + ext}")
    else:
        print(f"  [DRY  ] {name}  →  {new_stem + ext}")
    return 1


def rename_in_dir(directory: str, plan: dict, exts: set, apply: bool) -> int:
    """Rename matching slide files in `directory` (flat + one nested level)."""
    if not os.path.isdir(directory):
        print(f"  [SKIP] Directory not found: {directory}")
        return 0
    count = 0
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if entry.is_file():
            count += _rename_file(entry.path, entry.name, directory, plan, exts, apply)
        elif entry.is_dir():
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_file():
                    count += _rename_file(
                        sub.path, sub.name, entry.path, plan, exts, apply
                    )
    return count


def _swap_path_stem(file_location, plan: dict):
    """Replace the basename stem of a path with its new name (dir + ext kept).

    Returns the path unchanged when the stem is not in `plan` or is missing.
    """
    if pd.isna(file_location):
        return file_location
    s = str(file_location)
    old_stem = extract_file_name(s)
    new_stem = plan.get(old_stem)
    if new_stem is None:
        return file_location
    # Keep original separators by splitting on the last one.
    parts = re.split(r"([\\/])", s)
    base = parts[-1]
    _, ext = os.path.splitext(base.split("?")[0].split("#")[0])
    parts[-1] = new_stem + ext
    return "".join(parts)


def rewrite_metadata_csvs(plan: dict, iga_slides_csv: str, registry_csv: str,
                          out_dir: str, apply: bool):
    """Write copies of the source metadata CSVs with slide names updated.

    IgA      : 'File Location' basenames are swapped to the new names.
    Registry : 'ANON_name' values are swapped to the new names.
    Copies are written into out_dir under their original filenames; downstream
    tasks point --iga_slides_csv / --registry_csv at these copies.
    """
    if apply:
        os.makedirs(out_dir, exist_ok=True)

    for src, swap in (
        (iga_slides_csv, lambda df: df.assign(
            **{"File Location": df["File Location"].apply(
                lambda p: _swap_path_stem(p, plan))})),
        (registry_csv, lambda df: df.assign(
            ANON_name=df["ANON_name"].apply(
                lambda s: plan.get(str(s).strip(), s)))),
    ):
        if not os.path.isfile(src):
            print(f"  [SKIP] Metadata CSV not found: {src}")
            continue
        df = pd.read_csv(src)
        before = df.copy()
        df = swap(df)
        # Count rows whose slide column actually changed.
        col = "File Location" if "File Location" in df.columns else "ANON_name"
        n_changed = int((df[col].astype(str) != before[col].astype(str)).sum())
        dest = os.path.join(out_dir, os.path.basename(src))
        if apply:
            df.to_csv(dest, index=False)
            print(f"  [CSV  ] {os.path.basename(src)}: {n_changed} rows updated → {dest}")
        else:
            print(f"  [DRY  ] {os.path.basename(src)}: {n_changed} rows would update → {dest}")


def run_rename(args):
    """Entry point for --rename mode."""
    mode_label = "APPLY" if args.apply else "DRY RUN"
    print(f"Rename mode: {mode_label}\n")

    exts = {e if e.startswith(".") else f".{e}" for e in args.slide_exts}

    # --- Collect every known slide stem and its biopsy number ----------------
    records: dict = {}        # old_stem -> biopsy_number (CSV is source of truth)
    all_known: set = set()    # every stem in use, for collision-free new names

    if os.path.isfile(args.iga_slides_csv):
        for stem, biopsy in build_iga_records(args.iga_slides_csv):
            records.setdefault(stem, biopsy)
            all_known.add(stem)
    else:
        print(f"[WARN ] IgA metadata CSV not found: {args.iga_slides_csv}")

    if os.path.isfile(args.registry_csv):
        for stem, biopsy in build_registry_records(args.registry_csv):
            records.setdefault(stem, biopsy)
            all_known.add(stem)
    else:
        print(f"[WARN ] Registry metadata CSV not found: {args.registry_csv}")

    # Slides present on disk but absent from the CSVs get an unknown biopsy.
    for d in args.slide_dirs:
        for stem in discover_stems(d, exts):
            all_known.add(stem)
            records.setdefault(stem, "")

    # --- Build the rename plan -----------------------------------------------
    plan = assign_new_names(records, all_known, args.rename_all)
    if not plan:
        print(
            "No slides need renaming (all names are already safe). "
            "Pass --rename-all to force a remap of every slide."
        )
        return

    n_unknown = sum(1 for s in plan if not records.get(s))
    print(f"{len(plan)} slide(s) to rename  ({n_unknown} with unknown biopsy).\n")

    # --- Outputs --------------------------------------------------------------
    write_mapping_csv(plan, records, args.mapping_out, args.apply)

    total = 0
    for d in args.slide_dirs:
        print(f"--- Renaming in: {d} ---")
        n = rename_in_dir(d, plan, exts, args.apply)
        total += n
        print(f"  Summary: {n} file(s) {'renamed' if args.apply else 'to rename'}\n")

    print("--- Rewriting metadata CSV copies ---")
    rewrite_metadata_csvs(
        plan, args.iga_slides_csv, args.registry_csv, args.csv_out_dir, args.apply
    )

    print("\n" + "=" * 60)
    print(f"Slides mapped to safe names : {len(plan)}")
    print(f"Files {'renamed' if args.apply else 'to rename'} on disk      : {total}")
    print(f"Mapping CSV                 : {args.mapping_out}")
    print(f"Rewritten metadata CSV dir  : {args.csv_out_dir}")
    if not args.apply:
        print(
            "\nThis was a dry run — no files were changed.\n"
            "Run with --apply to write the mapping, rename files, and emit CSV copies."
        )


# ---------------------------------------------------------------------------
# Directory-rename mode: normalise biopsy-level subdirectory names
# ---------------------------------------------------------------------------


def rename_dirs_in(directory: str, apply: bool) -> int:
    """Rename first-level subdirectories of `directory` to normalised form.

    Each subdirectory name is passed through transform_label() then
    biopsy_to_dirname() (e.g. "B2312" → "12-23"). When the target name already
    exists as a sibling, the two are merged (items moved across, source removed).
    Returns the count of directories (to be) renamed.
    """
    if not os.path.isdir(directory):
        print(f"  [SKIP] Directory not found: {directory}")
        return 0
    count = 0
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        new_name = biopsy_to_dirname(transform_label(entry.name))
        if new_name == entry.name:
            continue
        new_path = os.path.join(directory, new_name)
        merge = os.path.isdir(new_path)
        if apply:
            tag = "MERGE" if merge else "REN  "
        else:
            tag = "DRY  "
        print(f"  [{tag}] {entry.name!r:30s} → {new_name!r}")
        if apply:
            if merge:
                # Destination already exists — move every item across, then
                # remove the now-empty source directory.
                for item in os.scandir(entry.path):
                    shutil.move(item.path, os.path.join(new_path, item.name))
                os.rmdir(entry.path)
            else:
                os.rename(entry.path, new_path)
        count += 1
    return count


def run_rename_dirs(args):
    """Entry point for --rename_dirs mode."""
    mode_label = "APPLY" if args.apply else "DRY RUN"
    print(f"Directory-rename mode: {mode_label}\n")

    total = 0
    for d in args.slide_dirs:
        print(f"--- {d} ---")
        n = rename_dirs_in(d, args.apply)
        total += n
        print(f"  Summary: {n} dir(s) {'renamed' if args.apply else 'to rename'}\n")

    print("=" * 60)
    print(f"Total director{'y' if total == 1 else 'ies'} "
          f"{'renamed' if args.apply else 'to rename'}: {total}")
    if not args.apply and total:
        print("\nThis was a dry run — no directories were changed.\n"
              "Run with --apply to perform the renames.")


# ---------------------------------------------------------------------------
# Core: reorganize one directory
# ---------------------------------------------------------------------------


def reorganize_dir(directory: str, mapping: dict, apply: bool) -> tuple:
    """Move flat slide files into biopsy subdirectories.

    Only processes files directly under `directory` (non-recursive). Files
    already inside a subdirectory are treated as already nested and skipped.

    Returns (moved, already_nested, unmapped) counts.
    """
    if not os.path.isdir(directory):
        print(f"  [SKIP] Directory not found: {directory}")
        return 0, 0, 0

    moved = already_nested = unmapped = 0

    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if entry.is_dir():
            already_nested += 1
            continue
        if not entry.is_file():
            continue

        stem, ext = os.path.splitext(entry.name)
        if ext not in SLIDE_EXTS:
            continue

        biopsy_dir = mapping.get(stem)
        if biopsy_dir is None:
            print(
                f"  [WARN ] No biopsy mapping for '{stem}' in {directory}"
                f" — will move to 'unknown/'"
            )
            biopsy_dir = "unknown"
            unmapped += 1

        dest_dir = os.path.join(directory, biopsy_dir)
        dest_path = os.path.join(dest_dir, entry.name)

        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(entry.path, dest_path)
            print(f"  [MOVE ] {entry.name}  →  {biopsy_dir}/{entry.name}")
        else:
            print(f"  [DRY  ] {entry.name}  →  {biopsy_dir}/{entry.name}")

        moved += 1

    return moved, already_nested, unmapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # The status lines use the '→' glyph; force UTF-8 so output does not crash
    # when stdout is piped/redirected on Windows (default cp1252 can't encode it).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Reorganize flat WSI slide directories into biopsy-nested layout.\n"
            "Default: dry run (no files moved). Pass --apply to execute."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iga_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directories that hold IgA slides at the top level "
            "(feats / labels / coords). Biopsy mapping comes from --iga_slides_csv."
        ),
    )
    parser.add_argument(
        "--registry_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Directories that hold registry slides (IgA_registry + non_IgA) at the "
            "top level. Biopsy mapping comes from --registry_csv."
        ),
    )
    parser.add_argument(
        "--iga_slides_csv",
        default="followup_data/derived/renamed/IgA_full_data.csv",
        help="IgA slide metadata CSV (columns: Biopsy Number, File Location).",
    )
    parser.add_argument(
        "--registry_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
        help="Registry metadata CSV (columns: ANON_name, biop_number).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the actual file moves. Omitting this flag is a safe dry run.",
    )

    # --- Rename mode ----------------------------------------------------------
    parser.add_argument(
        "--rename",
        action="store_true",
        help=(
            "Rename mode: assign filesystem-safe sequential names to slides with "
            "unsafe names (spaces / punctuation), write a mapping CSV, rename the "
            "files in --slide_dirs, and emit rewritten metadata CSV copies. "
            "Ignores --iga_dirs / --registry_dirs (those drive the move mode)."
        ),
    )
    parser.add_argument(
        "--rename_dirs",
        action="store_true",
        help=(
            "Directory-rename mode: normalise the biopsy-level subdirectory names "
            "of each --slide_dirs entry via transform_label() + biopsy_to_dirname() "
            "(e.g. 'B2312' → '12-23'), merging into an existing sibling when the "
            "normalised name already exists. Renames directories, not files."
        ),
    )
    parser.add_argument(
        "--slide_dirs",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "Rename / rename_dirs mode: directories to operate on. In --rename, the "
            "slide files within (feature / label / coord / raw-WSI dirs, flat or "
            "biopsy-nested) are renamed; in --rename_dirs, their first-level "
            "subdirectories are renamed."
        ),
    )
    parser.add_argument(
        "--slide_exts",
        nargs="*",
        default=[".npy", ".h5"],
        metavar="EXT",
        help="Rename mode: file extensions treated as slides (e.g. .npy .h5 .svs).",
    )
    parser.add_argument(
        "--rename_all",
        action="store_true",
        help="Rename mode: remap every slide, not just those with unsafe names.",
    )
    parser.add_argument(
        "--mapping_out",
        default="followup_data/derived/renamed/slide_name_mapping.csv",
        help="Rename mode: output path for the biopsy_number/old_name/new_name CSV.",
    )
    parser.add_argument(
        "--csv_out_dir",
        default="followup_data/derived/renamed",
        help=(
            "Rename mode: directory for rewritten metadata CSV copies "
            "(downstream tasks point --iga_slides_csv / --registry_csv here)."
        ),
    )

    args = parser.parse_args()

    if args.rename and args.rename_dirs:
        parser.error("Use only one of --rename or --rename_dirs.")

    if args.rename_dirs:
        if not args.slide_dirs:
            parser.error("--rename_dirs requires at least one --slide_dirs entry.")
        run_rename_dirs(args)
        return

    if args.rename:
        if not args.slide_dirs:
            parser.error("--rename requires at least one --slide_dirs entry.")
        run_rename(args)
        return

    if not args.iga_dirs and not args.registry_dirs:
        parser.error("Provide at least one of --iga_dirs or --registry_dirs.")

    mode_label = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode_label}\n")

    # --- Build mappings -------------------------------------------------------
    iga_mapping: dict = {}
    if args.iga_dirs:
        print(f"Loading IgA slide mapping from: {args.iga_slides_csv}")
        iga_mapping = build_iga_mapping(args.iga_slides_csv)
        print(f"  {len(iga_mapping)} IgA slides mapped to biopsy directories.\n")

    registry_mapping: dict = {}
    if args.registry_dirs:
        print(f"Loading registry mapping from: {args.registry_csv}")
        registry_mapping = build_registry_mapping(args.registry_csv)
        print(
            f"  {len(registry_mapping)} registry slides mapped to biopsy directories.\n"
        )

    # --- Process directories --------------------------------------------------
    total_moved = total_nested = total_unmapped = 0

    for d in args.iga_dirs:
        print(f"--- IgA: {d} ---")
        m, n, u = reorganize_dir(d, iga_mapping, args.apply)
        total_moved += m
        total_nested += n
        total_unmapped += u
        print(
            f"  Summary: {m} files {'moved' if args.apply else 'to move'}"
            f" | {n} already nested (skipped) | {u} unmapped (→ unknown/)\n"
        )

    for d in args.registry_dirs:
        print(f"--- Registry: {d} ---")
        m, n, u = reorganize_dir(d, registry_mapping, args.apply)
        total_moved += m
        total_nested += n
        total_unmapped += u
        print(
            f"  Summary: {m} files {'moved' if args.apply else 'to move'}"
            f" | {n} already nested (skipped) | {u} unmapped (→ unknown/)\n"
        )

    print("=" * 60)
    print(f"Total files {'moved' if args.apply else 'to move'} : {total_moved}")
    print(f"Total already-nested (skipped)     : {total_nested}")
    print(f"Total unmapped (→ unknown/)         : {total_unmapped}")
    if not args.apply:
        print(
            "\nThis was a dry run — no files were changed.\n"
            "Run with --apply to perform the actual moves."
        )


if __name__ == "__main__":
    main()
