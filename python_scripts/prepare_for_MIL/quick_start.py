#!/usr/bin/env python
"""
Quick Start — TRIDENT WSI feature-extraction pipeline.

Interactively drives the full TRIDENT-based pipeline:
    1. Tissue segmentation        (stock TRIDENT, run once over all slides)
    2. Patch coordinates          (stock TRIDENT, run once over all slides)
    3. Patch feature extraction   (per-stain stain-normalised, run_trident_stain_feats.py)
    4. Reorganise into the biopsy-nested {backbone}_feats/{biopsy_nr}/{slide_id}.h5 layout
       consumed by ProcessedMILDataset (reorganize_trident_feats.py)

Steps 3–4 reproduce the CLAM-path output format (one .h5 per slide with 'features' +
'coords'), so downstream MIL training is unchanged.

Run this with the TRIDENT environment's Python (the one where `pip install -e .` was run in
python_scripts/external_repositories/TRIDENT-main):

    python python_scripts/prepare_for_MIL/quick_start.py

Every step prints the exact command it runs, so you can also copy/paste and run stages
manually. Stain normalisation is optional: if you skip it, stock TRIDENT `--task feat` is
used on raw patches instead.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trident_io import coords_dir_name, features_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRIDENT_RUNNER = ROOT / "python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py"
STAIN_FEATS = ROOT / "python_scripts/prepare_for_MIL/run_trident_stain_feats.py"
REORG = ROOT / "python_scripts/prepare_for_MIL/reorganize_trident_feats.py"

# Friendly name -> TRIDENT encoder name, encoder input patch_size (px), and the output
# feats-dir name matching the CLAM-path convention. Default input is 224x224 for every
# encoder (the native ViT resolution). TRIDENT's table suggests 256 for UNI2-h; we
# standardize on 224 for a consistent feature-extraction input — override at the prompt.
BACKBONES = {
    "1": {"name": "UNI2-h",      "trident": "uni_v2",   "patch_size": 224, "feats_dir": "UNI2-h_feats"},
    "2": {"name": "Virchow2",    "trident": "virchow2", "patch_size": 224, "feats_dir": "Virchow2_feats"},
    "3": {"name": "H-optimus-1", "trident": "hoptimus1","patch_size": 224, "feats_dir": "h-optimus-1_feats"},
    "4": {"name": "Hibou-L",     "trident": "hibou_l",  "patch_size": 224, "feats_dir": "Hibou-L_feats"},
}


def hr(char="=", n=70):
    print(char * n)


def ask(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    resp = input(f"{msg}{suffix}: ").strip()
    return resp or (default if default is not None else "")


def ask_yes(msg, default="y"):
    return (ask(msg + " (y/n)", default).lower() or default) == "y"


def run_cmd(cmd, desc):
    """Run a subprocess, streaming output. Returns True on success."""
    print(f"\n🎬 {desc}")
    print("   $ " + " ".join(str(c) for c in cmd) + "\n")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode == 0:
        print(f"\n✅ {desc} — done")
        return True
    print(f"\n❌ {desc} — failed (exit {result.returncode})")
    return False


def collect_failed_slides(job_dir):
    """Scan TRIDENT per-stage logs ('_logs_*.txt') for failed slides.

    With --skip_errors, TRIDENT records each failure as a line
    '<slide_filename>: ERROR: <message>' in a per-stage log under job_dir (and in
    wsi_states/). Returns a list of (stage_log_name, slide_filename, message).
    """
    failures = []
    for root, _dirs, files in os.walk(job_dir):
        for name in files:
            if name.startswith("_logs_") and name.endswith(".txt"):
                try:
                    with open(
                        os.path.join(root, name), "r", encoding="utf-8", errors="replace"
                    ) as f:
                        for line in f:
                            if ": ERROR:" in line:
                                slide, _sep, msg = line.partition(": ")
                                failures.append((name, slide.strip(), msg.strip()))
                except OSError:
                    continue
    return failures


def write_error_report(job_dir):
    """Aggregate TRIDENT failure logs into {job_dir}/quick_start_errors.log.

    Returns (failures, report_path). The report is overwritten each call so it always
    reflects the latest attempt per slide (TRIDENT logs are keyed per slide).
    """
    failures = collect_failed_slides(job_dir)
    report_path = os.path.join(job_dir, "quick_start_errors.log")
    if failures:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# {len(failures)} slide(s) failed across stages\n")
            for stage_log, slide, msg in failures:
                f.write(f"[{stage_log}] {slide}: {msg}\n")
    return failures, report_path


def report_errors(cfg, stage_label):
    """Print a one-line running tally of failed slides after a stage."""
    failures, report_path = write_error_report(cfg["job_dir"])
    if failures:
        print(
            f"\n⚠️  {len(failures)} slide(s) errored so far (after {stage_label}); "
            f"continuing. Filenames logged → {report_path}"
        )


def check_environment():
    """Warn if TRIDENT is not importable in the current interpreter."""
    print("\n✅ Using Python:", sys.executable)
    if importlib.util.find_spec("trident") is None:
        print(
            "\n⚠️  `trident` is not importable in this interpreter.\n"
            "   Activate the TRIDENT venv and install it first:\n"
            "     cd python_scripts/external_repositories/TRIDENT-main\n"
            "     pip install -e .\n"
            "   then re-run this script with that venv's Python."
        )
        if not ask_yes("\nContinue anyway?", "n"):
            sys.exit(1)
    else:
        print("✅ `trident` is importable")


def collect_config():
    """Prompt for the run configuration."""
    hr("-")
    print("Configuration")
    hr("-")

    cfg = {}
    cfg["wsi_dir"] = ask("Raw WSI directory", "data/raw_wsi/IgA")
    cfg["job_dir"] = ask("TRIDENT job/output directory", "WSI/IgA/trident")
    cfg["search_nested"] = ask_yes("Are WSIs in biopsy-nested subfolders?", "y")
    cfg["skip_errors"] = ask_yes("Skip slides that error (continue + log filename)?", "y")
    cfg["segmenter"] = ask("Segmenter (hest/grandqc/otsu)", "hest")
    cfg["mag"] = float(ask("Target magnification", "20"))
    cfg["overlap"] = int(ask("Patch overlap (px)", "0"))
    cfg["gpu"] = int(ask("GPU index (-1 for CPU)", "0"))
    cfg["batch_size"] = int(ask("Feature-extraction batch size", "256"))

    print("\nBackbone:")
    for key, b in BACKBONES.items():
        star = " 🌟" if key == "1" else ""
        print(f"   [{key}] {b['name']} (TRIDENT: {b['trident']}, patch {b['patch_size']}px){star}")
    choice = ask("Choose backbone number", "1")
    cfg["backbone"] = BACKBONES.get(choice, BACKBONES["1"])
    cfg["patch_size"] = int(ask("Patch size (px)", str(cfg["backbone"]["patch_size"])))

    # Single source of truth for biopsy nesting (file_name,biopsy_number) and for
    # the per-stain extractor's stain column — used by both step_features and step_reorganize.
    cfg["labels_csv"] = ask("Labels CSV (file_name,biopsy_number,stain[,mpp])", "label_csvs/labels_unfiltered.csv")

    cfg["use_stain"] = ask_yes("Apply per-stain stain normalisation?", "y")
    if cfg["use_stain"]:
        cfg["stain_refs_dir"] = ask("Stain references dir (.pt per stain)", "stain_refs/IgA")

    cfg["coords_dir"] = coords_dir_name(cfg["mag"], cfg["patch_size"], cfg["overlap"])
    cfg["features_subdir"] = features_dir(
        cfg["job_dir"], cfg["coords_dir"], cfg["backbone"]["trident"]
    )
    cfg["reorg_output"] = ask(
        "Reorganised output dir (biopsy-nested)",
        os.path.join(os.path.dirname(cfg["job_dir"]) or ".", cfg["backbone"]["feats_dir"]),
    )
    return cfg


def step_segmentation(cfg):
    cmd = [
        sys.executable, TRIDENT_RUNNER,
        "--task", "seg",
        "--wsi_dir", cfg["wsi_dir"],
        "--job_dir", cfg["job_dir"],
        "--segmenter", cfg["segmenter"],
        "--gpus", str(cfg["gpu"]),
    ]
    if cfg["search_nested"]:
        cmd.append("--search_nested")
    if cfg["skip_errors"]:
        cmd.append("--skip_errors")
    return run_cmd(cmd, "Step 1/4 — tissue segmentation")


def step_coords(cfg):
    cmd = [
        sys.executable, TRIDENT_RUNNER,
        "--task", "coords",
        "--wsi_dir", cfg["wsi_dir"],
        "--job_dir", cfg["job_dir"],
        "--mag", f"{cfg['mag']:g}",
        "--patch_size", str(cfg["patch_size"]),
        "--overlap", str(cfg["overlap"]),
    ]
    if cfg["search_nested"]:
        cmd.append("--search_nested")
    if cfg["skip_errors"]:
        cmd.append("--skip_errors")
    return run_cmd(cmd, "Step 2/4 — patch coordinates")


def step_features(cfg):
    if cfg["use_stain"]:
        cmd = [
            sys.executable, STAIN_FEATS,
            "--wsi_dir", cfg["wsi_dir"],
            "--job_dir", cfg["job_dir"],
            "--labels_csv", cfg["labels_csv"],
            "--stain_refs_dir", cfg["stain_refs_dir"],
            "--backbone", cfg["backbone"]["trident"],
            "--mag", f"{cfg['mag']:g}",
            "--patch_size", str(cfg["patch_size"]),
            "--overlap", str(cfg["overlap"]),
            "--batch_size", str(cfg["batch_size"]),
            "--gpu_index", str(cfg["gpu"]),
        ]
        if cfg["search_nested"]:
            cmd.append("--search_nested")
        if cfg["skip_errors"]:
            cmd.append("--skip_errors")
        return run_cmd(cmd, "Step 3/4 — per-stain feature extraction")

    # Plain stock TRIDENT feature extraction (raw patches, no stain normalisation).
    cmd = [
        sys.executable, TRIDENT_RUNNER,
        "--task", "feat",
        "--wsi_dir", cfg["wsi_dir"],
        "--job_dir", cfg["job_dir"],
        "--patch_encoder", cfg["backbone"]["trident"],
        "--mag", f"{cfg['mag']:g}",
        "--patch_size", str(cfg["patch_size"]),
        "--overlap", str(cfg["overlap"]),
        "--feat_batch_size", str(cfg["batch_size"]),
        "--gpus", str(cfg["gpu"]),
    ]
    if cfg["search_nested"]:
        cmd.append("--search_nested")
    if cfg["skip_errors"]:
        cmd.append("--skip_errors")
    return run_cmd(cmd, "Step 3/4 — feature extraction (no stain norm)")


def step_reorganize(cfg):
    cmd = [
        sys.executable, REORG,
        "--features_dir", cfg["features_subdir"],
        "--labels_csv", cfg["labels_csv"],
        "--output_dir", cfg["reorg_output"],
        "--copy",  # keep the original TRIDENT output in place
    ]
    return run_cmd(cmd, "Step 4/4 — reorganise into biopsy-nested layout")


def main():
    hr()
    print("🔱  TRIDENT WSI FEATURE EXTRACTION — QUICK START")
    hr()

    check_environment()
    cfg = collect_config()

    hr("-")
    print("Plan")
    hr("-")
    print(f"  WSI dir         : {cfg['wsi_dir']}")
    print(f"  Job dir         : {cfg['job_dir']}")
    print(f"  Backbone        : {cfg['backbone']['name']} ({cfg['backbone']['trident']})")
    print(f"  Mag / patch     : {cfg['mag']:g}x / {cfg['patch_size']}px / {cfg['overlap']}px overlap")
    print(f"  Stain norm      : {'on' if cfg['use_stain'] else 'off'}")
    print(f"  Skip errors     : {'on' if cfg['skip_errors'] else 'off'}")
    print(f"  Features        : {cfg['features_subdir']}")
    print(f"  Reorg output    : {cfg['reorg_output']}")

    if not ask_yes("\nProceed?", "y"):
        print("Aborted.")
        return 0

    steps = [
        ("segmentation", step_segmentation),
        ("coords", step_coords),
        ("features", step_features),
        ("reorganize", step_reorganize),
    ]
    for name, fn in steps:
        if not ask_yes(f"\nRun {name} step?", "y"):
            print(f"   ⏭️  skipped {name}")
            continue
        ok = fn(cfg)
        # Surface per-slide failures recorded by TRIDENT (skip_errors continues the run,
        # so the stage still 'succeeds' overall while individual slides are logged).
        if name in ("segmentation", "coords", "features"):
            report_errors(cfg, name)
        if not ok:
            print(f"\n❌ Pipeline stopped at '{name}'. Fix the error above and re-run "
                  "(completed slides are skipped automatically).")
            return 1

    failures, report_path = write_error_report(cfg["job_dir"])

    hr()
    print("✅ PIPELINE COMPLETE")
    hr()
    if failures:
        print(f"\n⚠️  {len(failures)} slide(s) failed and were skipped. "
              f"Full list → {report_path}")
        for _stage_log, slide, _msg in failures:
            print(f"     - {slide}")
        print()
    print(
        f"""
Features ready for MIL training at:
  {cfg['reorg_output']}/{{biopsy_nr}}/{{slide_id}}.h5   (keys: 'features', 'coords')

Next: point your MIL training script (e.g. regression_MIL.py) at this directory via
--features_paths. See README.md for the MIL / survival commands.
"""
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
