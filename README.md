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


Example run:
# Tiling
python python_scripts/prepare_for_MIL/run_clam_tiling.py --wsi_dir data/raw_wsi --output_dir WSI/IgA --step_size 112 --stitch store_false

# Reorganize and rename
python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --iga_dirs WSI/IgA/patches --apply

python python_scripts/prepare_for_MIL/reorganize_wsi_dirs.py --rename --slide_dirs WSI/IgA/patches data/unlabeled/IgA --apply

# Labels definition
python python_scripts/prepare_for_MIL/define_labels.py --iga_output_dir WSI/IgA/labels --iga_date_filter None

python python_scripts/prepare_for_MIL/define_regression_labels.py --iga_output_dir WSI/IgA/labels_regression --iga_date_filter None

# Stain normalization (Vahadane or Macenko)
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/patches --wsi_dir data/unlabeled/IgA --output stain_refs/IgA --labels_csv label_csvs/labels_unfiltered.csv --save_patches stain_refs/IgA/qc --n_save_patches 20 --method macenko

# Feature extraction
python python_scripts/prepare_for_MIL/compute_feats_clam.py --patches_dir WSI/IgA/patches --wsi_dir data/unlabeled/IgA --output_dir WSI/IgA/UNI2-h_feats --backbone UNI2-h --batch_size 128 --labels_csv label_csvs/labels_unfiltered.csv --stain_refs_dir stain_refs/IgA

# MIL
python python_scripts/MIL/regression_MIL.py  --model_type transmil --features_paths WSI/IgA/UNI2-h_feats --labels_paths WSI/IgA/labels_regression --checkpoint_dir checkpoints_regression_transmil_CLAM --log_dir results/losses_regression_transmil --dropout 0.1 --save_every 5 --batch_size 1

# Attention map
python python_scripts/MIL/visualize_attention.py --features_paths WSI/IgA/UNI2-h_feats --checkpoint checkpoints_regression_transmil_CLAM/transmil_regression.pth --model_type transmil --task regression --label_csv label_csvs/labels_IgA.csv

# Survival curve
python python_scripts/MIL/evaluate_survival.py --model_type deepgraphsurv --checkpoint checkpoints_CLAM/deepgraphsurv_model.pth --features_paths WSI/IgA/UNI2-h_feats --labels_paths WSI/IgA/labels --val_csv validation_files_csvs/survival_validation_20pct_both.csv --dropout 0.1