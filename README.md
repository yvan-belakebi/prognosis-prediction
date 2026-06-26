# Structure

1) Tiling
2) Feature extraction
3) MIL pooling
4) Survival model

The studied events are:
- End-stage kidney disease (eGFR < 15mL/min/1,73m²): 1
- Death: 2
- Transplant: 3
- Dialysis: 4
- No event before censoring: 0

### Feature extraction

The foundation models tested for feature extraction are: Virchow2, UNI2-h, Hibou-L, Hibou-B, H-optimus-1.


torchMIL, CLAM, StainStyleSampler and torchstain should be downloaded and included in the python_scripts folder.


Latest version now uses the TRIDENT repository, also from Mahmood lab

1) Installation

sudo dnf install python3.11-devel
python3.11 -m venv .trident_venv
pip install -r requirements.txt
cd python_scripts/external_repositories/TRIDENT-main/
pip install -e .
cd ../torchmil
pip install -e .
cd /data/yvan-files/prognosis_prediction
If running on vm without internet access, the errors will guide you to allow you to run locally.
In python_scripts/external_repositories/TRIDENT-main/trident/patch_encoder_models/local_ckpts.json change the paths to your local installation of the models, e.g. "hoptimus1": "/data/yvan-files/prognosis-prediction/models/hoptimus1/pytorch_model.bin",

2) Quick start

python python_scripts/prepare_for_MIL/quick_start.py

OR run the stages manually (equivalent to quick_start.py). `--search_nested` is needed because
the raw WSIs are stored in biopsy-nested subfolders. The TRIDENT job dir (WSI/IgA/trident) holds
segmentation, coordinates and raw feature output; the final step folds features into the
biopsy-nested layout (WSI/IgA/UNI2-h_feats) consumed by the MIL stage.

2) Tiling  (tissue segmentation + patch coordinates, TRIDENT)

# Tissue segmentation (HEST; use --segmenter otsu for a weights-free CPU fallback)
python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task seg --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --segmenter hest --gpus 0 --search_nested

# Patch coordinates (20x, 224 px, no overlap)
python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task coords --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --mag 20 --patch_size 224 --overlap 0 --search_nested (--dump_patches)

# Labels definition
python python_scripts/prepare_for_MIL/define_labels.py --iga_output_dir WSI/IgA/trident/labels --iga_date_filter None

python python_scripts/prepare_for_MIL/define_regression_labels.py --iga_output_dir WSI/IgA/trident/labels_regression --iga_date_filter None

3) Feature extraction  (TRIDENT patch encoder, per-stain stain-normalized)

# (Optional) Stain references — one .pt per stain, used by the per-stain extractor below.
# fit_stain_reference.py auto-detects TRIDENT coords (point --patches_dir at the job's
# patches/ dir); read recipe is detected per-file from the h5 attrs. Omit to skip stain
# normalization and use stock TRIDENT --task feat (raw patches) instead.
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_macenko/IgA --labels_csv label_csvs/labels_unfiltered.csv --save_patches stain_refs_macenko/IgA/qc --n_save_patches 20 --method macenko --skip_errors

python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_vahadane/IgA --labels_csv label_csvs/labels_unfiltered.csv --save_patches stain_refs_vahadane/IgA/qc --n_save_patches 20 --method vahadane --skip_errors

After having run both, create a stain_refs dir and copy paste the relevant .pt file for each stain
(Explicit choice of the method).
Recommended: Vahadane for Lendrum, Masson Goldner, Masson trichrom, PASM, HES, MSB
# Per-stain, stain-normalized feature extraction (backbone names: uni_v2, virchow2, hoptimus1, hibou_l)
python python_scripts/prepare_for_MIL/run_trident_stain_feats.py --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --labels_csv label_csvs/labels_unfiltered.csv --stain_refs_dir stain_refs/IgA --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 --batch_size 256 --search_nested

# (Optional) Speed benchmark: Macenko vs Vahadane normalization on a small subset.
# Fit one reference set per method first (stored under stain_refs_macenko / stain_refs_vahadane),
# then time run_trident_stain_feats.py's extraction path both ways on --n_slides slides.
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_macenko/IgA  --labels_csv label_csvs/labels_unfiltered.csv --method macenko
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_vahadane/IgA --labels_csv label_csvs/labels_unfiltered.csv --method vahadane
python python_scripts/prepare_for_MIL/benchmark_stain_norm.py --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --labels_csv label_csvs/labels_unfiltered.csv --macenko_refs_dir stain_refs_macenko/IgA --vahadane_refs_dir stain_refs_vahadane/IgA --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 --n_slides 4 --batch_size 256 --search_nested

#   Without stain normalization, use stock TRIDENT instead:
#   python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task feat --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --patch_encoder uni_v2 --mag 20 --patch_size 224 --overlap 0 --gpus 0 --search_nested

# Reorganize flat TRIDENT features into the biopsy-nested layout (recovers biopsy_nr from the raw WSI dir)
python python_scripts/prepare_for_MIL/reorganize_trident_feats.py --features_dir WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --wsi_dir data/raw_wsi/IgA --output_dir WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested --copy

Or actually:

python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --iga_dirs WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --mapping_out followup_data/derived/renamed/slide_name_mapping_trident.csv --slide_dirs WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested --apply

4) MIL pooling
python python_scripts/MIL/regression_MIL.py  --model_type transmil --features_paths WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2_biopsy_nested --labels_paths WSI/IgA/trident/labels_regression --checkpoint_dir checkpoints_regression_transmil --log_dir results/losses_regression_transmil --dropout 0.1 --save_every 5 --batch_size 1

# Attention map
python python_scripts/MIL/visualize_attention.py --features_paths WSI/IgA/UNI2-h_feats --checkpoint checkpoints_regression_transmil/transmil_regression.pth --model_type transmil --task regression --label_csv label_csvs/labels_IgA.csv

5) Survival model
python python_scripts/MIL/evaluate_survival.py --model_type deepgraphsurv --checkpoint checkpoints/deepgraphsurv_model.pth --features_paths WSI/IgA/UNI2-h_feats --labels_paths WSI/IgA/labels --val_csv validation_files_csvs/survival_validation_20pct_both.csv --dropout 0.1


---

### Legacy CLAM pipeline (pre-TRIDENT)

Kept for reference. Tiling + feature extraction via CLAM instead of TRIDENT:

# Tiling
python python_scripts/prepare_for_MIL/run_clam_tiling.py --wsi_dir data/raw_wsi --output_dir WSI/IgA --step_size 112 --stitch store_false

# Reorganize and rename
python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --iga_dirs WSI/IgA/patches --apply
python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename --slide_dirs WSI/IgA/patches data/unlabeled/IgA --apply

# Stain normalization (Vahadane or Macenko)
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/patches --wsi_dir data/unlabeled/IgA --output stain_refs/IgA --labels_csv label_csvs/labels_unfiltered.csv --save_patches stain_refs/IgA/qc --n_save_patches 20 --method macenko

# Feature extraction
python python_scripts/prepare_for_MIL/compute_feats_clam.py --patches_dir WSI/IgA/patches --wsi_dir data/unlabeled/IgA --output_dir WSI/IgA/UNI2-h_feats --backbone UNI2-h --batch_size 128 --labels_csv label_csvs/labels_unfiltered.csv --stain_refs_dir stain_refs/IgA