# Complete Project Structure & File Guide

## 🏗️ Project Directory Layout

```
prognosis prediction/
│
├── 📋 Documentation (NEW - Created)
│   ├── PROJECT_COMPLETION.md          ← Start here for overview
│   ├── WSI_PIPELINE_README.md         ← Complete pipeline guide
│   ├── PATCH_EXTRACTION_GUIDE.md      ← Detailed extraction guide
│   ├── MODIFICATIONS_SUMMARY.md       ← Technical changes
│   └── DIRECTORY_STRUCTURE.md         ← This file
│
├── 🐍 Python Scripts
│   └── prepare_for_MIL/
│       ├── 🔧 Modified Files
│       │   ├── deepzoom_tiler.py      ← MODIFIED: Now reads from data/raw_wsi/
│       │   └── extract_features.py    ← MODIFIED: Added torch import
│       │
│       ├── 🆕 New Utilities
│       │   ├── test_pipeline.py       ← Validation script
│       │   ├── quick_start.py         ← Interactive setup wizard
│       │   └── load_feature_extractor.py
│       │
│       ├── 📊 Feature Extraction
│       │   └── compute_feats.py       ← Uses output from deepzoom_tiler
│       │
│       └── 🎯 Existing Utilities
│           └── load_feature_extractor.py
│
├── 📊 Data Directory
│   ├── raw_wsi/                       ← Input: .svs and .ndpi files
│   │   └── test_patient/
│   │       └── 2013_220016_ANON.svs   ← Test data provided
│   │
│   └── followup_data/
│       └── IgA_cohort_full_data.csv
│
├── 👥 Patient Lists
│   ├── patient_lists/
│   │   ├── Patient_list_for_follow_up_data.csv
│   │   ├── patient_list.csv
│   │   └── relevant_patients.py
│
├── 📦 Configuration Files
│   ├── requirements.txt               ← All dependencies
│   └── .venv/                         ← Virtual environment
│
└── 📋 Exploration Scripts
    └── explore_data/
        └── patient_data.py
```

## 📂 Output Directory Structure (Created During Runtime)

### After Patch Extraction
```
WSI/
└── prognosis/
    └── single/                        ← Single magnification (default)
        ├── 2013_220016_ANON/
        │   ├── 2013_220016_ANON_0_0.jpeg
        │   ├── 2013_220016_ANON_0_1.jpeg
        │   ├── 2013_220016_ANON_1_0.jpeg
        │   └── ... (hundreds of patches)
        │
        └── other_slide/
            ├── other_slide_0_0.jpeg
            └── ... (patches)
```

### Alternative: Hierarchical Output
```
WSI/
└── prognosis/
    └── pyramid/                       ← Multi-magnification
        ├── 2013_220016_ANON/
        │   ├── low_mag_0_0.jpeg
        │   ├── low_mag_0_0/
        │   │   ├── high_mag_0_0.jpeg
        │   │   ├── high_mag_0_1.jpeg
        │   │   └── ... (high mag patches)
        │   └── ... (more low mag patches)
        │
        └── other_slide/
            └── ... (similar structure)
```

### After Feature Extraction
```
datasets/
└── prognosis/
    ├── 2013_220016_ANON.csv
    │   └── Feature vectors (1536 dims for UNI2-h)
    │
    ├── other_slide.csv
    │   └── Feature vectors
    │
    └── ... (one CSV per slide)
```

## 📖 File-by-File Guide

### 🔴 Critical Files (Must Read First)

#### `PROJECT_COMPLETION.md`
- ✅ What was completed
- ✅ File inventory
- ✅ Test results
- ✅ Quick start instructions
- **Read if**: You want quick overview

#### `WSI_PIPELINE_README.md`
- ✅ Architecture diagrams
- ✅ Quick start (5 minutes)
- ✅ Data format reference
- ✅ Parameter tables
- **Read if**: You want complete understanding

### 🟠 Implementation Files (Read if Modifying)

#### `deepzoom_tiler.py` (MODIFIED)
- 📍 **Lines Modified**: ~80 lines
- 📍 **Key Changes**:
  - Added: Input directory configuration
  - Added: .ndpi file support
  - Changed: Output directory structure
  - Added: Logging
  - Added: Error handling
- **Read if**: You want to understand patch extraction

#### `extract_features.py` (MODIFIED)
- 📍 **Lines Modified**: 1 line added
- 📍 **Change**: Added `import torch`
- **Read if**: You want complete details

### 🟡 Utility Files (Use as Needed)

#### `test_pipeline.py` (NEW)
- 🎯 Purpose: Validate pipeline setup
- 📝 Usage: `python test_pipeline.py`
- **Use if**: You get errors or need validation

#### `quick_start.py` (NEW)
- 🎯 Purpose: Interactive guided setup
- 📝 Usage: `python quick_start.py`
- **Use if**: First time using the pipeline

### 🟢 Documentation Files (Reference as Needed)

#### `PATCH_EXTRACTION_GUIDE.md`
- 📚 200+ lines of detailed usage
- 📍 Step-by-step instructions
- 📍 All command options explained
- 📍 Troubleshooting section
- **Read if**: You need detailed guidance

#### `MODIFICATIONS_SUMMARY.md`
- 📚 150+ lines technical summary
- 📍 Before/after comparison
- 📍 Compatibility matrix
- 📍 Data flow diagram
- **Read if**: You're a developer/researcher

#### `DIRECTORY_STRUCTURE.md` (This File)
- 📚 Navigation guide
- 📍 File purposes
- 📍 What to read when
- **Read if**: You're exploring the project

## 🔄 Reading Path by Use Case

### Case 1: Just Want to Extract & Compute Features
1. Read: `PROJECT_COMPLETION.md`
2. Run: `quick_start.py`
3. Follow interactive prompts

### Case 2: Want Full Understanding
1. Read: `WSI_PIPELINE_README.md`
2. Read: `PATCH_EXTRACTION_GUIDE.md`
3. Read: `MODIFICATIONS_SUMMARY.md`
4. Run: `test_pipeline.py`

### Case 3: Troubleshooting Issues
1. Run: `test_pipeline.py` (identify problem)
2. Check: Specific section in `PATCH_EXTRACTION_GUIDE.md`
3. Look: `MODIFICATIONS_SUMMARY.md` for changes
4. Read: Docstrings in code files

### Case 4: Modifying Code
1. Read: `MODIFICATIONS_SUMMARY.md` (understand changes)
2. Read: Docstrings in `deepzoom_tiler.py`
3. Check: `test_pipeline.py` (validation)
4. Verify: With your modifications

### Case 5: Setting Up in New Environment
1. Copy: All files to new location
2. Read: `PROJECT_COMPLETION.md`
3. Run: `quick_start.py`
4. Install: Dependencies as prompted

## 📊 Configuration Files Reference

### `requirements.txt`
Contains all Python dependencies. Key packages:
- `openslide-python`: WSI reading
- `torch`: Feature extraction
- `timm`: Model loading
- `pandas`: Data handling
- `scikit-image`: Image processing
- `Pillow`: Image I/O

### Default Argument Values (Quick Reference)

**deepzoom_tiler.py**
```
--dataset prognosis
--input_dir data/raw_wsi
--tile_size 224
--base_mag 20
--objective 40
--background_t 15
--workers 4
--format jpeg
--quality 70
--magnifications 0
--overlap 0
```

**compute_feats.py**
```
--dataset prognosis
--backbone UNI2-h           (or h-optimus-1, Virchow2)
--batch_size 128
--num_workers 4
--magnification single       (or high, low, tree)
```

## 🎯 Quick Navigation Table

| Want to... | File to Read | Command to Run |
|-----------|--------------|----------------|
| Get started | PROJECT_COMPLETION.md | python quick_start.py |
| Extract patches | PATCH_EXTRACTION_GUIDE.md | python deepzoom_tiler.py |
| Compute features | PATCH_EXTRACTION_GUIDE.md | python compute_feats.py |
| Validate setup | MODIFICATIONS_SUMMARY.md | python test_pipeline.py |
| Understand arch | WSI_PIPELINE_README.md | (read only) |
| Modify code | Source files + docstrings | (edit then test) |
| Find parameters | PATCH_EXTRACTION_GUIDE.md | (reference) |
| Troubleshoot | PATCH_EXTRACTION_GUIDE.md | python test_pipeline.py |

## 🔐 Important Notes

### File Dependencies
```
deepzoom_tiler.py
  ├── Depends on: openslide
  ├── Creates: WSI/prognosis/single/
  └── Used by: compute_feats.py

compute_feats.py
  ├── Depends on: deepzoom_tiler output
  ├── Creates: datasets/prognosis/
  ├── Loads: load_feature_extractor.py
  └── Outputs: Ready for MIL training

extract_features.py
  ├── Utility function
  └── Used by: compute_feats.py (optional)
```

### Read-Only Files
- ✅ All .md documentation files (safe to read)
- ✅ test_pipeline.py (only validates, no changes)
- ✅ All original python_scripts/ files (except mentioned modified)

### Files You Should NOT Modify
- ❌ data/raw_wsi/ (input only)
- ❌ requirements.txt (unless adding new dependencies)
- ❌ WSI/ directory (automatically generated)
- ❌ datasets/ directory (automatically generated)

### Safe to Modify
- ✅ deepzoom_tiler.py (if understanding changes)
- ✅ compute_feats.py (if adding features)
- ✅ Any script in python_scripts/prepare_for_MIL/

## 📈 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Apr 17, 2026 | Initial modifications complete |
| | | - Modified deepzoom_tiler.py |
| | | - Fixed extract_features.py |
| | | - Added 5 documentation files |
| | | - Added test_pipeline.py |
| | | - Added quick_start.py |

## 🎓 Learning Resources

### For Understanding WSI Processing
- `WSI_PIPELINE_README.md` → Architecture section
- `PATCH_EXTRACTION_GUIDE.md` → Data format section

### For Understanding Feature Extraction
- `load_feature_extractor.py` → Code comments
- `compute_feats.py` → BagDataset and compute_feats function

### For Understanding MIL Training
- `PATCH_EXTRACTION_GUIDE.md` → Next steps section
- Comment in feature CSV files about expected format

---

**Last Updated**: April 17, 2026
**Status**: Complete and Production-Ready ✅
