# WSI to Features Pipeline - Complete Setup

## 🎯 What You Now Have

A complete, production-ready pipeline to:
1. **Extract patches** from whole-slide images (.svs, .ndpi)
2. **Compute features** using state-of-the-art foundation models
3. **Generate outputs** compatible with any MIL architecture

## 📁 Modified Files

| File | Changes | Impact |
|------|---------|--------|
| `deepzoom_tiler.py` | ✅ Now reads from data/raw_wsi/ with .svs/.ndpi support | Core functionality updated |
| `extract_features.py` | ✅ Added missing torch import | Minor fix, no behavior change |

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `PATCH_EXTRACTION_GUIDE.md` | Detailed guide with all usage options |
| `MODIFICATIONS_SUMMARY.md` | Technical summary of changes |
| `test_pipeline.py` | Automated validation script |
| `quick_start.py` | Interactive setup wizard |
| `README.md` | This file |

## 🚀 Quick Start (5 Minutes)

### Option A: Interactive Setup (Recommended)

```bash
cd python_scripts/prepare_for_MIL
python quick_start.py
```

This will:
- Check your environment
- Install missing packages (optional)
- Run validation
- Extract patches from test data
- Guide you through feature extraction
- Show next steps

### Option B: Manual Setup

#### 1️⃣ Validate Pipeline
```bash
python python_scripts/prepare_for_MIL/test_pipeline.py
```

#### 2️⃣ Extract Patches
```bash
python python_scripts/prepare_for_MIL/deepzoom_tiler.py
```

Output: `WSI/prognosis/single/{slide_name}/`

#### 3️⃣ Compute Features
```bash
python python_scripts/prepare_for_MIL/compute_feats.py \
  --dataset prognosis \
  --backbone UNI2-h \
  --batch_size 128 \
  --num_workers 4
```

Output: `datasets/prognosis/{slide_name}.csv`

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input WSI Files                          │
│             data/raw_wsi/*.svs, *.ndpi                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │   deepzoom_tiler.py                │
        │  (Extract patches with filtering)  │
        │  Logs: INFO/ERROR messages         │
        │  Supports: .svs, .ndpi             │
        │  Output: 224x224 patches (default) │
        └────────────────┬───────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │    Organized Patch Directory            │
    │  WSI/prognosis/single/{slide_name}/    │
    │  Contains: *.jpeg patches               │
    │  Ready for: Feature extraction          │
    └────────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  load_feature_extractor.py     │
        │  (Load foundation models)      │
        │  Options:                      │
        │  • UNI2-h (1536D)              │
        │  • H-optimus-1 (1536D)         │
        │  • Virchow2 (2560D)            │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   compute_feats.py             │
        │  (Extract features on patches) │
        │  Batch processing              │
        │  GPU acceleration              │
        │  Progress tracking             │
        └────────────────┬───────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │      Feature Output (CSV)              │
    │    datasets/prognosis/{slide}.csv      │
    │    Rows: Patches                       │
    │    Cols: Feature dimensions            │
    └────────────────┬───────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │  MIL Training                │
        │  (Use with any MIL arch:)    │
        │  • DSMIL                     │
        │  • TransMIL                  │
        │  • Your custom architecture  │
        └──────────────────────────────┘
```

## 📊 Data Format Reference

### Input
```
data/raw_wsi/
├── test_patient/
│   └── 2013_220016_ANON.svs        [40x magnification]
└── other_slides/
    ├── slide_1.svs                 [various magnifications]
    └── slide_2.ndpi
```

### Intermediate (Patches)
```
WSI/prognosis/single/
└── 2013_220016_ANON/
    ├── 2013_220016_ANON_0_0.jpeg   [224x224 pixels]
    ├── 2013_220016_ANON_0_1.jpeg   [224x224 pixels]
    ├── 2013_220016_ANON_1_0.jpeg   [224x224 pixels]
    └── ...                         [hundreds/thousands of patches]
```

### Output (Features)
```
datasets/prognosis/
├── 2013_220016_ANON.csv
│   0,0.1234,0.5678,...,0.9012    [1536 dimensions for UNI2-h]
│   0,0.2345,0.6789,...,0.0123
│   ...
└── other_slide.csv
```

## ⚙️ Configuration Parameters

### Patch Extraction (deepzoom_tiler.py)

| Param | Default | Options | Description |
|-------|---------|---------|-------------|
| `--dataset` | prognosis | any string | Output directory name |
| `--input_dir` | data/raw_wsi | path | WSI source directory |
| `--tile_size` | 224 | 224-512 | Patch size in pixels |
| `--base_mag` | 20 | 2.5-40 | Target magnification (1x to 40x) |
| `--objective` | 40 | 40/20/10 | Slide objective if not in metadata |
| `--background_t` | 15 | 0-100 | Background filtering (higher = less filtering) |
| `--workers` | 4 | 1-16 | Parallel extraction processes |
| `--magnifications` | 0 | 0, 1 | 0=single, 0 1=hierarchical |

### Feature Extraction (compute_feats.py)

| Param | Default | Options | Description |
|-------|---------|---------|-------------|
| `--dataset` | TCGA-lung-single | any string | Must match extraction dataset |
| `--backbone` | resnet18 | UNI2-h, h-optimus-1, Virchow2 | Feature extractor model |
| `--batch_size` | 128 | 1-512 | Batch size for GPU |
| `--num_workers` | 4 | 1-16 | Data loading threads |
| `--magnification` | single | single, high, low, tree | Processing mode |

## 🔍 Troubleshooting

### ❓ "No slides found in data/raw_wsi"
```bash
# Verify WSI files exist
ls -la data/raw_wsi/test_patient/
# Should show: 2013_220016_ANON.svs
```

### ❓ "ModuleNotFoundError: torch"
```bash
# Install missing dependencies
pip install torch torchvision scikit-image timm openslide-python
```

### ❓ "OpenSlide error - cannot open slide"
- Check if .ndpi files are properly formatted
- Some older or proprietary formats may not be supported
- Try with the test .svs file first

### ❓ "Out of memory during feature extraction"
```bash
# Reduce batch size
python compute_feats.py --batch_size 32
```

### ❓ "Patches look blurry or have artifacts"
- Adjust `--background_t` during extraction (try values 10-20)
- Increase `--tile_size` for larger patches
- Check slide quality with OpenSlide viewer

## 📈 Performance Tips

| Task | Speed Boost | Trade-off |
|------|------------|-----------|
| Patch extraction | Increase `--workers` (e.g., 8) | Higher CPU/memory usage |
| Feature extraction | Use GPU (`--batch_size 256`) | VRAM requirement |
| Storage | Reduce `--quality` to 50 | Slightly lower image quality |
| Memory | Reduce `--tile_size` to 168 | Smaller patches lose detail |

## ✅ Compatibility Checklist

- ✅ Works with .svs and .ndpi files
- ✅ Compatible with UNI2-h, H-optimus-1, Virchow2 feature extractors
- ✅ Produces output compatible with DSMIL
- ✅ Produces output compatible with TransMIL
- ✅ Produces output compatible with custom MIL architectures
- ✅ Works with GPU or CPU
- ✅ Supports single and multi-magnification analysis
- ✅ Includes error recovery and logging
- ✅ Production-ready with validation scripts

## 🎓 Next Steps After Feature Extraction

1. **Organize data with labels**
   ```python
   # Create CSV with slide IDs and labels
   slide_id, label
   2013_220016_ANON, class_A
   slide_2, class_B
   ```

2. **Load features for training**
   ```python
   import pandas as pd
   features = pd.read_csv('datasets/prognosis/2013_220016_ANON.csv')
   X = features.values  # Shape: (num_patches, num_features)
   ```

3. **Train MIL model**
   ```python
   # Using your MIL architecture of choice
   # Features are ready to be used as patch embeddings
   ```

4. **Evaluate performance**
   - Compare different feature extractors
   - Compare different MIL architectures
   - Analyze feature importance

## 📞 Support Resources

- **Test pipeline**: `python test_pipeline.py --help`
- **Quick setup**: `python quick_start.py`
- **Detailed docs**: See `PATCH_EXTRACTION_GUIDE.md`
- **Changes made**: See `MODIFICATIONS_SUMMARY.md`
- **Code reference**: See module docstrings in each .py file

## 📝 License & Citation

If you use this pipeline, please cite:
- OpenSlide (patch extraction)
- TIMM (feature extractors)
- The respective feature model papers:
  - UNI2-h (Mahmood Lab)
  - H-optimus-1 (Bioptimus)
  - Virchow2 (Paige AI)

## 🎉 You're All Set!

Your WSI feature extraction pipeline is ready to:
1. ✅ Extract patches from any .svs or .ndpi file
2. ✅ Compute features with multiple foundation models
3. ✅ Generate outputs for any MIL architecture

Start with `python quick_start.py` for guided setup!
