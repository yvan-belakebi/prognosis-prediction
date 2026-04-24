# WSI Patch Extraction and Feature Computation Guide

## Overview

This guide explains how to extract patches from whole-slide images (WSI) and compute features using different feature extractors compatible with multiple MIL architectures.

## Pipeline Components

### 1. **deepzoom_tiler.py** (Modified)
Extracts patches from .svs and .ndpi files in `data/raw_wsi/` and organizes them for feature extraction.

**Features:**
- Reads .svs and .ndpi files from `data/raw_wsi/` (including subdirectories)
- Performs automatic background filtering
- Supports single and hierarchical (multi-magnification) tiling
- Outputs patches in directories compatible with `compute_feats.py`
- Logging for tracking progress and debugging

**Default output structure:**
```
WSI/prognosis/single/
  ├── 2013_220016_ANON/
  │   ├── patch_0_0.jpeg
  │   ├── patch_0_1.jpeg
  │   └── ...
```

### 2. **load_feature_extractor.py**
Provides feature extractors from state-of-the-art models:
- **UNI2-h**: 1536-dimensional features
- **H-optimus-1**: 1536-dimensional features
- **Virchow2**: 2560-dimensional features

### 3. **extract_features.py**
Simple wrapper for feature extraction on individual images.

### 4. **compute_feats.py**
Main orchestration script that:
- Loads patches for each WSI
- Extracts features using selected backbone
- Saves features as CSV files for MIL training

## Usage

### Step 1: Extract Patches from WSI Files

Run the tiling script with default settings:

```bash
python python_scripts/prepare_for_MIL/deepzoom_tiler.py
```

**Optional arguments:**
```bash
python python_scripts/prepare_for_MIL/deepzoom_tiler.py \
  --dataset prognosis \                 # Dataset name (default: prognosis)
  --input_dir data/raw_wsi \            # WSI source directory (default: data/raw_wsi)
  --tile_size 224 \                     # Patch size (default: 224)
  --base_mag 20 \                       # Target magnification (default: 20)
  --objective 40 \                      # Slide objective (default: 40)
  --background_t 15 \                   # Background filtering threshold (default: 15)
  --workers 4 \                         # Number of worker processes (default: 4)
  --format jpeg \                       # Patch format (default: jpeg)
  --quality 70                          # JPEG quality (default: 70)
```

**For hierarchical extraction (multi-magnification):**
```bash
python python_scripts/prepare_for_MIL/deepzoom_tiler.py \
  --magnifications 0 1                  # Extract at 2 magnification levels
```

This will create patches in `WSI/prognosis/pyramid/` instead of `WSI/prognosis/single/`

### Step 2: Compute Features

#### Using foundation models (recommended):

**With UNI2-h (1536D features):**
```bash
python python_scripts/prepare_for_MIL/compute_feats.py \
  --dataset prognosis \
  --backbone UNI2-h \
  --batch_size 128 \
  --num_workers 4 \
  --magnification single
```

**With H-optimus-1 (1536D features):**
```bash
python python_scripts/prepare_for_MIL/compute_feats.py \
  --dataset prognosis \
  --backbone h-optimus-1 \
  --batch_size 128 \
  --num_workers 4 \
  --magnification single
```

**With Virchow2 (2560D features):**
```bash
python python_scripts/prepare_for_MIL/compute_feats.py \
  --dataset prognosis \
  --backbone Virchow2 \
  --batch_size 128 \
  --num_workers 4 \
  --magnification single
```

**Output files:**
Features are saved to `datasets/prognosis/*/` as CSV files where each row is a patch's feature vector.

### Step 3: Train MIL Models

The extracted features are now ready for MIL training with various architectures (DSMIL, TransMIL, etc.)

## Data Format and Compatibility

### Input Format (raw_wsi/)
- **Supported formats**: .svs, .ndpi
- **Location**: `data/raw_wsi/` (can be in subdirectories)
- **Example**: `data/raw_wsi/test_patient/2013_220016_ANON.svs`

### Intermediate Format (WSI/)
After patch extraction, organized as:

**Single magnification:**
```
WSI/prognosis/single/
  └── {patientID}/
      ├── {patientID}_0_0.jpeg
      ├── {patientID}_0_1.jpeg
      └── ...
```

**Hierarchical (multi-magnification):**
```
WSI/prognosis/pyramid/
  └── {patientID}/
      ├── {low_mag_patch_0_0}.jpeg
      ├── {low_mag_patch_0_0}/
      │   ├── {high_mag_patch_0_0}.jpeg
      │   ├── {high_mag_patch_0_1}.jpeg
      │   └── ...
      └── ...
```

### Feature Output Format (datasets/)
```
datasets/prognosis/
  └── {slide_name}.csv
```

CSV format: Each row is a patch, columns are feature dimensions (e.g., 1536 columns for UNI2-h)

## Magnification Options

The script supports different magnifications during extraction:

| Target Mag | Argument | Description |
|-----------|----------|-------------|
| 5x | `--magnifications 0` | Extracted at 5x (40/8 from 40x slide) |
| 10x | `--magnifications 0` | Default: 10x (40/4 from 40x slide) |
| 20x | `--magnifications 0` | 20x (40/2 from 40x slide) |
| Multi-level | `--magnifications 0 1` | Hierarchical with 2 magnification levels |

The actual magnification depends on:
1. Slide's objective power (detected from metadata or `--objective` arg)
2. `--base_mag` parameter (target magnification)

## Troubleshooting

### Issue: "No .svs or .ndpi files found"
**Solution**: Ensure your WSI files are in `data/raw_wsi/` or its subdirectories

### Issue: "Error loading slide"
**Solution**: Verify the slide file is valid and not corrupted. OpenSlide may not support all WSI formats.

### Issue: Memory errors during patch extraction
**Solution**: Reduce `--workers` or increase system RAM

### Issue: Poor background filtering
**Solution**: Adjust `--background_t` (lower = more aggressive filtering, higher = less filtering)

## Next Steps

After computing features:
1. Organize feature CSVs with clinical labels
2. Train MIL models using the feature vectors
3. Evaluate on test set using different MIL architectures
4. Compare performance across feature extractors

## Important Notes

- ✅ **Compatible with all downstream MIL architectures** - The patch organization follows standard conventions
- ✅ **Supports multiple feature extractors** - UNI2-h, H-optimus-1, Virchow2
- ✅ **GPU acceleration** - Feature extraction uses GPU when available
- ✅ **Multiprocessing** - Patch extraction uses multiprocessing for speed
- ✅ **Error handling** - Gracefully handles corrupted slides and continues processing

## Performance Notes

- **Patch extraction**: ~10-30 minutes per slide (depends on size and workers)
- **Feature computation**: ~2-5 minutes per slide with UNI2-h on GPU
- For 40x slides: Default tile_size=224 at 20x magnification creates patches of ~44 microns
