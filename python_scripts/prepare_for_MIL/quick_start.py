#!/usr/bin/env python
"""
Quick Start Guide for WSI Feature Extraction Pipeline

Run this script to get started with the complete pipeline:
1. Extract patches from WSI files
2. Compute features using foundation models
3. Get output ready for MIL training

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_venv():
    """Check if we're in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing required packages...\n")

    packages = [
        "openslide-python",
        "torch",
        "torchvision",
        "scikit-image",
        "timm",
        "huggingface_hub",
    ]

    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"], capture_output=True
        )
        if result.returncode != 0:
            print(f"⚠️  Failed to install {package}")
            print(result.stderr.decode())
        else:
            print(f"✅ {package} installed")

    print()


def run_extraction():
    """Run patch extraction on test data."""
    print("\n🎬 Starting patch extraction from data/raw_wsi/...\n")

    result = subprocess.run(
        [sys.executable, "python_scripts/prepare_for_MIL/deepzoom_tiler.py"],
        capture_output=False,
        text=True,
    )

    if result.returncode == 0:
        print("\n✅ Patch extraction completed successfully!")
        return True
    else:
        print("\n❌ Patch extraction failed")
        return False


def show_feature_options():
    """Show feature extraction options."""
    print("\n" + "=" * 70)
    print("🔬 FEATURE EXTRACTION OPTIONS")
    print("=" * 70)

    options = {
        "1": {
            "name": "UNI2-h (Recommended)",
            "dims": "1536D",
            "command": """python python_scripts/prepare_for_MIL/compute_feats.py \\
  --dataset prognosis \\
  --backbone UNI2-h \\
  --batch_size 128 \\
  --num_workers 4 \\
  --magnification single""",
        },
        "2": {
            "name": "H-optimus-1",
            "dims": "1536D",
            "command": """python python_scripts/prepare_for_MIL/compute_feats.py \\
  --dataset prognosis \\
  --backbone h-optimus-1 \\
  --batch_size 128 \\
  --num_workers 4 \\
  --magnification single""",
        },
        "3": {
            "name": "Virchow2",
            "dims": "2560D",
            "command": """python python_scripts/prepare_for_MIL/compute_feats.py \\
  --dataset prognosis \\
  --backbone Virchow2 \\
  --batch_size 128 \\
  --num_workers 4 \\
  --magnification single""",
        },
    }

    print("\nChoose a feature extractor:")
    print()

    for key, option in options.items():
        marker = "🌟" if key == "1" else "  "
        print(f"{marker} [{key}] {option['name']} ({option['dims']})")

    print("\nExample command (UNI2-h):")
    print(options["1"]["command"])

    print("\n" + "=" * 70)


def check_extraction_output():
    """Check if patch extraction output exists."""
    expected_dirs = [
        "WSI/prognosis/single",
        "WSI/prognosis",
    ]

    for directory in expected_dirs:
        if os.path.exists(directory):
            # Count patches
            patches = []
            for root, dirs, files in os.walk(directory):
                patches.extend([f for f in files if f.endswith((".jpg", ".jpeg"))])

            if patches:
                print(f"\n✅ Found {len(patches)} patches in {directory}/")
                return True

    return False


def main():
    print("\n" + "=" * 70)
    print("🚀 WSI FEATURE EXTRACTION - QUICK START")
    print("=" * 70)

    # Check virtual environment
    if not check_python_venv():
        print("\n⚠️  You appear to be using system Python")
        print("   It's recommended to use a virtual environment:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows")
    else:
        print("\n✅ Using Python virtual environment")

    # Step 1: Installation
    print("\n" + "-" * 70)
    print("Step 1: Install Dependencies")
    print("-" * 70)

    response = (
        input("\nInstall/update required packages? (y/n) [y]: ").strip().lower() or "y"
    )
    if response == "y":
        install_dependencies()

    # Step 2: Validation
    print("\n" + "-" * 70)
    print("Step 2: Validate Pipeline")
    print("-" * 70)

    result = subprocess.run(
        [sys.executable, "python_scripts/prepare_for_MIL/test_pipeline.py"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0 and "Missing packages" not in result.stdout:
        print("⚠️  Some validation checks failed")

    # Step 3: Extract patches
    print("\n" + "-" * 70)
    print("Step 3: Extract Patches from WSI")
    print("-" * 70)

    response = (
        input("\nExtract patches from data/raw_wsi/? (y/n) [y]: ").strip().lower()
        or "y"
    )
    if response == "y":
        if run_extraction():
            if check_extraction_output():
                print("\n✅ Patches ready for feature extraction!")
            else:
                print("⚠️  Could not verify patch extraction output")
        else:
            print("❌ Patch extraction failed")
            return 1

    # Step 4: Feature extraction options
    print("\n" + "-" * 70)
    print("Step 4: Compute Features")
    print("-" * 70)

    show_feature_options()

    response = input("\nCompute features now? (y/n) [n]: ").strip().lower()
    if response == "y":
        backbone = (
            input(
                "Enter backbone number (1=UNI2-h, 2=h-optimus-1, 3=Virchow2) [1]: "
            ).strip()
            or "1"
        )

        backbones = {
            "1": "UNI2-h",
            "2": "h-optimus-1",
            "3": "Virchow2",
        }

        if backbone in backbones:
            print(f"\n🔬 Computing features with {backbones[backbone]}...\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "python_scripts/prepare_for_MIL/compute_feats.py",
                    "--dataset",
                    "prognosis",
                    "--backbone",
                    backbones[backbone],
                    "--batch_size",
                    "128",
                    "--num_workers",
                    "4",
                ],
                capture_output=False,
            )

            if result.returncode == 0:
                print("\n✅ Feature extraction completed!")
                # Show output location
                print("\n📊 Features saved to: datasets/prognosis/")
                print("   Ready for MIL training!")
            else:
                print("\n❌ Feature extraction failed")
                return 1

    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE SETUP COMPLETE")
    print("=" * 70)

    print(
        """
Your WSI features are now ready for training!

Next steps:
1. Organize your data with labels
2. Choose a MIL architecture (DSMIL, TransMIL, etc.)
3. Train your model on the extracted features
4. Evaluate on test set

For more details, see:
  - PATCH_EXTRACTION_GUIDE.md
  - MODIFICATIONS_SUMMARY.md

Questions? Check test_pipeline.py --help
    """
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
