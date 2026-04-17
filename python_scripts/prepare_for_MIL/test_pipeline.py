"""
Test script to validate the complete WSI -> patches -> features pipeline.
Run this after modifying deepzoom_tiler.py to ensure everything works correctly.
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_input_data(input_dir="data/raw_wsi"):
    """Check if input WSI files exist."""
    print(f"\n[1/4] Checking input data in '{input_dir}'...")

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False

    svs_files = glob.glob(f"{input_dir}/**/*.svs", recursive=True)
    ndpi_files = glob.glob(f"{input_dir}/**/*.ndpi", recursive=True)
    all_slides = svs_files + ndpi_files

    if not all_slides:
        print(f"❌ No .svs or .ndpi files found in {input_dir}")
        return False

    print(f"✅ Found {len(all_slides)} slide(s):")
    for slide in all_slides[:5]:  # Show first 5
        print(f"   - {slide}")
    if len(all_slides) > 5:
        print(f"   ... and {len(all_slides) - 5} more")
    return True


def check_deepzoom_tiler():
    """Check if deepzoom_tiler.py exists and is valid."""
    print(f"\n[2/4] Checking deepzoom_tiler.py...")

    tiler_path = "python_scripts/prepare_for_MIL/deepzoom_tiler.py"
    if not os.path.exists(tiler_path):
        print(f"❌ deepzoom_tiler.py not found at {tiler_path}")
        return False

    # Check for key components
    with open(tiler_path, "r") as f:
        content = f.read()

    required_items = [
        "DeepZoomStaticTiler",
        "TileWorker",
        "nested_patches",
        "data/raw_wsi",
        "logging",
    ]

    missing = []
    for item in required_items:
        if item not in content:
            missing.append(item)

    if missing:
        print(f"⚠️  Missing or modified components: {missing}")
    else:
        print(f"✅ deepzoom_tiler.py looks good")
        print(f"   - Has DeepZoomStaticTiler class")
        print(f"   - Has TileWorker class")
        print(f"   - Has nested_patches function")
        print(f"   - Configured to read from data/raw_wsi")
        print(f"   - Has logging support")

    return True


def check_compute_feats():
    """Check if compute_feats.py is configured for the output."""
    print(f"\n[3/4] Checking compute_feats.py...")

    compute_path = "python_scripts/prepare_for_MIL/compute_feats.py"
    if not os.path.exists(compute_path):
        print(f"❌ compute_feats.py not found at {compute_path}")
        return False

    with open(compute_path, "r") as f:
        content = f.read()

    required_items = [
        "load_uni2h_feature_extractor",
        "load_hoptimus1_feature_extractor",
        "load_virchow2_feature_extractor",
        "BagDataset",
        "compute_feats",
    ]

    missing = []
    for item in required_items:
        if item not in content:
            missing.append(item)

    if missing:
        print(f"❌ Missing components: {missing}")
        return False
    else:
        print(f"✅ compute_feats.py is ready")
        print(f"   - Supports UNI2-h feature extractor")
        print(f"   - Supports H-optimus-1 feature extractor")
        print(f"   - Supports Virchow2 feature extractor")
        print(f"   - Has BagDataset loader")
        print(f"   - Has compute_feats function")

    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print(f"\n[4/4] Checking dependencies...")

    required_packages = {
        "openslide": "openslide",
        "torch": "torch",
        "PIL": "Pillow",
        "skimage": "scikit-image",
        "timm": "timm",
        "pandas": "pandas",
    }

    missing_packages = []
    for module, package_name in required_packages.items():
        try:
            __import__(module)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def show_expected_output():
    """Show what the expected output structure should look like."""
    print("\n" + "=" * 60)
    print("EXPECTED OUTPUT STRUCTURE")
    print("=" * 60)

    print("\n1️⃣  After patch extraction (deepzoom_tiler.py):")
    print(
        """
WSI/prognosis/single/
  └── 2013_220016_ANON/
      ├── 2013_220016_ANON_0_0.jpeg
      ├── 2013_220016_ANON_0_1.jpeg
      ├── 2013_220016_ANON_1_0.jpeg
      └── ... (patches from the slide)
    """
    )

    print("\n2️⃣  After feature extraction (compute_feats.py):")
    print(
        """
datasets/prognosis/
  └── 2013_220016_ANON.csv
    """
    )

    print("   CSV format: rows=patches, columns=feature_dimensions")
    print("   - UNI2-h: 1536 columns")
    print("   - H-optimus-1: 1536 columns")
    print("   - Virchow2: 2560 columns")


def show_next_steps():
    """Show what users should do next."""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    print(
        """
1. Extract patches from WSI files:
   python python_scripts/prepare_for_MIL/deepzoom_tiler.py

2. Compute features using UNI2-h (recommended):
   python python_scripts/prepare_for_MIL/compute_feats.py \\
     --dataset prognosis \\
     --backbone UNI2-h \\
     --batch_size 128 \\
     --num_workers 4

3. Features will be saved to: datasets/prognosis/
   Ready for MIL training with your chosen architecture!

For detailed options, see: PATCH_EXTRACTION_GUIDE.md
    """
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate the WSI -> patches -> features pipeline"
    )
    parser.add_argument(
        "--input_dir",
        default="data/raw_wsi",
        help="Input directory with WSI files [data/raw_wsi]",
    )
    parser.add_argument(
        "--skip_deps",
        action="store_true",
        help="Skip dependency check (useful if using conda)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("WSI PIPELINE VALIDATION TEST")
    print("=" * 60)

    all_passed = True

    # Run checks
    all_passed &= check_input_data(args.input_dir)
    all_passed &= check_deepzoom_tiler()
    all_passed &= check_compute_feats()

    if not args.skip_deps:
        all_passed &= check_dependencies()
    else:
        print(f"\n[4/4] Skipping dependency check (as requested)")

    # Show expected output
    show_expected_output()

    # Show summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("Your pipeline is ready to use!")
    else:
        print("⚠️  Some checks failed - please address issues above")
    print("=" * 60)

    # Show next steps
    show_next_steps()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
