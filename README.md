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


torchMIL, StainStyleSampler and torchstain should be downloaded and included in the python_scripts folder.


Latest version now uses the TRIDENT repository, also from Mahmood lab

1) Installation

Download trident and torchmil under the path specified in requirements.txt or change this path
sudo dnf install python3.11-devel
python3.11 -m venv .trident_venv
pip install -r requirements.txt
cd /data/yvan-files/prognosis_prediction
If running on vm without internet access, the errors will guide you to allow you to run locally.
In python_scripts/external_repositories/TRIDENT-main/trident/patch_encoder_models/local_ckpts.json change the paths to your local installation of the models, e.g. "hoptimus1": "/data/yvan-files/prognosis-prediction/models/hoptimus1/pytorch_model.bin",

2) Quick start

python python_scripts/prepare_for_MIL/quick_start.py

OR run the stages manually (equivalent to quick_start.py). `--search_nested` is needed because
the raw WSIs are stored in biopsy subfolders. The TRIDENT job dir (WSI/IgA/trident) holds
segmentation, coordinates and feature output. Features are written flat (one <slide>.h5 per slide)
and used as-is by the MIL stage: which biopsy, stain and train/val split a slide belongs to comes
from the labels CSV (step 3), never from folders.

2) Tiling  (tissue segmentation + patch coordinates, TRIDENT)

# Tissue segmentation (HEST; use --segmenter otsu for a weights-free CPU fallback)
python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task seg --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --segmenter hest --gpus 0 --search_nested

# Patch coordinates (20x, 224 px, no overlap)
python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task coords --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --mag 20 --patch_size 224 --overlap 0 --search_nested (--dump_patches)

3) Labels definition

One registry-style CSV (one row per slide) is the single data source. --query selects the rows
(any pandas query, e.g. "is_IgA == True" or "Biopsy_date >= '2006-01-01'"); the --*_col flags
name its columns (defaults match registry_anonymized.csv). Each script writes a flat folder of
<slide>.npy labels and a labels CSV (file_name, biopsy_number, patient, stain, label, split).
The train/val split is patient-level and stratified (--val_frac, default 0.2).
The input CSV must list the slides that were tiled: its slide column (--slide_col) is matched to
the feature file stems. registry_anonymized.csv covers the registry slides (IgA and non-IgA) but
not the separate IgA cohort, whose data would first have to be put into this format.

python python_scripts/prepare_for_MIL/define_labels.py --input_csv followup_data/derived/renamed/registry_anonymized.csv --query "is_IgA == True" --labels_dir WSI/IgA/trident/labels --labels_csv label_csvs/labels.csv

python python_scripts/prepare_for_MIL/define_regression_labels.py --input_csv followup_data/derived/renamed/registry_anonymized.csv --query "is_IgA == True" --labels_dir WSI/IgA/trident/labels_regression --labels_csv label_csvs/labels_regression.csv

4) Feature extraction  (TRIDENT patch encoder, per-stain stain-normalized)

# (Optional) Stain references — one .pt per stain, used by the per-stain extractor below.
# fit_stain_reference.py auto-detects TRIDENT coords (point --patches_dir at the job's
# patches/ dir); read recipe is detected per-file from the h5 attrs. Omit to skip stain
# normalization and use stock TRIDENT --task feat (raw patches) instead.
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_macenko/IgA --labels_csv label_csvs/labels.csv --save_patches stain_refs_macenko/IgA/qc --n_save_patches 20 --method macenko --skip_errors

python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_vahadane/IgA --labels_csv label_csvs/labels.csv --save_patches stain_refs_vahadane/IgA/qc --n_save_patches 20 --method vahadane --skip_errors

After having run both, create a stain_refs dir and copy paste the relevant .pt file for each stain
(Explicit choice of the method).
Recommended: Vahadane for Lendrum, Masson Goldner, Masson trichrom, PASM, HES, MSB
# Per-stain, stain-normalized feature extraction (backbone names: uni_v2, virchow2, hoptimus1, hibou_l)
python python_scripts/prepare_for_MIL/run_trident_stain_feats.py --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --labels_csv label_csvs/labels.csv --stain_refs_dir stain_refs/IgA --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 --batch_size 256 --search_nested

# (Optional) Speed benchmark: Macenko vs Vahadane normalization on a small subset.
# Fit one reference set per method first (stored under stain_refs_macenko / stain_refs_vahadane),
# then time run_trident_stain_feats.py's extraction path both ways on --n_slides slides.
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_macenko/IgA  --labels_csv label_csvs/labels.csv --method macenko
python python_scripts/prepare_for_MIL/fit_stain_reference.py --patches_dir WSI/IgA/trident/20x_224px_0px_overlap/patches --wsi_dir data/raw_wsi/IgA --output stain_refs_vahadane/IgA --labels_csv label_csvs/labels.csv --method vahadane
python python_scripts/prepare_for_MIL/benchmark_stain_norm.py --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --labels_csv label_csvs/labels.csv --macenko_refs_dir stain_refs_macenko/IgA --vahadane_refs_dir stain_refs_vahadane/IgA --backbone uni_v2 --mag 20 --patch_size 224 --overlap 0 --n_slides 4 --batch_size 256 --search_nested

#   Without stain normalization, use stock TRIDENT instead:
#   python python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py --task feat --wsi_dir data/raw_wsi/IgA --job_dir WSI/IgA/trident --patch_encoder uni_v2 --mag 20 --patch_size 224 --overlap 0 --gpus 0 --search_nested

5) MIL pooling

All training scripts take one --features_path (the flat TRIDENT features folder), one
--labels_path and the --labels_csv from step 3. --stains picks the stains to train on (default:
all; '|' merges source names, e.g. "HES|HE"; matching ignores case and punctuation). The
train/val split is the labels CSV's split column; --biopsy_sampling and --max_biopsies group
slides by its biopsy_number column.

python python_scripts/MIL/MIL.py --model_type abmil --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels --labels_csv label_csvs/labels.csv --stains PAS "HE|HES" --checkpoint_dir checkpoints --log_dir results/losses

python python_scripts/MIL/regression_MIL.py --model_type transmil --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels_regression --labels_csv label_csvs/labels_regression.csv --stains PAS --checkpoint_dir checkpoints_regression_transmil --log_dir results/losses_regression_transmil --dropout 0.1 --save_every 5 --batch_size 1

# Attention map (--labels_csv resolves biopsy IDs in --label_csv to a slide)
python python_scripts/evaluate_MIL/visualize_attention.py --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --checkpoint checkpoints_regression_transmil/transmil_regression.pth --model_type transmil --task regression --label_csv label_csvs/labels.csv --labels_csv label_csvs/labels.csv --authorized_slides_csv ok_slides.csv --save_as_tif

6) Survival model evaluation (val split of the labels CSV, risk averaged per biopsy)
python python_scripts/evaluate_MIL/evaluate_survival.py --model_type abmil --checkpoint checkpoints/abmil_model.pth --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels --labels_csv label_csvs/labels.csv --stains PAS

# Cohort survival (no model): Kaplan-Meier curve with one point per biopsy, plus patient/event
# counts, median follow-up and survival, and n-year survival with 95% CI (--split train|val|all)
python python_scripts/evaluate_MIL/plot_train_survival.py --labels_csv label_csvs/labels.csv --split all --years 1 3 5 10 --output_dir results/cohort_survival

7) Multi-stain fusion (UNICORN-style)

Predicts at the *biopsy* level instead of the slide level: the biopsy's slides are grouped by
stain, each stain is encoded by its own 2-layer transformer + attention pooling into one token,
a 2-layer transformer aggregates the stain tokens, and a single linear head outputs a Cox risk
score (--task survival) or a continuous target (--task regression). Stains a biopsy does not
have are masked out, so biopsies with different stain panels train together.

    python_scripts/MIL/multistain_data.py   biopsy-level, multi-stain bags
    python_scripts/MIL/multistain_model.py  MultiStainFusion (per-stain encoders + aggregator)
    python_scripts/MIL/multistain_MIL.py    training script (both tasks)

Inputs are the same as the single-stain scripts: the flat TRIDENT features, the .npy labels from
define_labels.py / define_regression_labels.py, and the labels CSV for each slide's biopsy, stain
and split. A stain argument may merge source names with '|' ("HES|HE"); matching ignores case and
punctuation.

# Survival
python python_scripts/MIL/multistain_MIL.py --task survival --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels --labels_csv label_csvs/labels.csv --stains PAS PASM "HES|HE" Congo AFOG --patches_per_stain 512 --batch_size 8 --epochs 50 --checkpoint_dir checkpoints_multistain --log_dir results/losses_multistain

# Regression (eGFR)
python python_scripts/MIL/multistain_MIL.py --task regression --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels_regression --labels_csv label_csvs/labels_regression.csv --stains PAS PASM "HES|HE" Congo AFOG --patches_per_stain 512 --batch_size 8 --epochs 50 --checkpoint_dir checkpoints_multistain_regression --log_dir results/losses_multistain_regression

# GPU memory scales as batch_size x n_stains x patches_per_stain x feat_dim; lower
# --patches_per_stain first if it does not fit. Prefer that over lowering --batch_size for
# survival: the Cox risk set is the micro-batch, so a small batch weakens the loss and
# --accumulation_steps does not compensate for it. --share_stain_encoder uses one patch
# encoder for all stains, which helps when the rarer stains cover few biopsies.
# The stain vocabulary and architecture are written to
# {checkpoint_dir}/multistain_config.json so evaluation can rebuild the same model.

# Stain-weight analysis — which stain drove which prediction, and can you trust it
python python_scripts/evaluate_MIL/analyze_stain_weights.py --checkpoint checkpoints_multistain/multistain_survival_best.pth --features_path WSI/IgA/trident/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/IgA/trident/labels --labels_csv label_csvs/labels.csv --split val --output_dir results/stain_analysis

# Reads the stain vocabulary and architecture from multistain_config.json next to the
# checkpoint, so the flags only describe the data. Writes, into --output_dir:
#   stain_contributions.csv          per biopsy: prediction, label, and each stain's
#                                    signed contribution / attention weight / lift
#   stain_contribution_heatmap.png   biopsies (sorted by prediction) x stains
#   stain_contribution_by_stain.png  cohort distribution per stain
#   stain_waterfall.png              three example biopsies, low / median / high
#   stain_ablation.csv/.png          leave-one-stain-out change in C-index (or MAE)
#   panel_baseline.csv               is the stain panel prognostic on its own?
#
# Report the *contribution*, not the attention weight: attention is renormalised over
# whichever stains a biopsy happens to have, so 0.33 means "uniform" in a 3-stain panel
# and "strongly upweighted" in a 10-stain one, and it carries no direction. The
# contribution is signed and reconstructs the prediction exactly (the script asserts
# this every batch). Use --lift_* in the CSV if you want the panel-size-corrected
# attention instead.
#
# Read panel_baseline.csv before making any claim about a stain: if the availability
# pattern alone is prognostic (Congo gets ordered when amyloid is suspected), part of
# what looks like stain importance is really panel-composition signal.
# The ablation is most meaningful when the model was trained with --stain_dropout > 0,
# which makes a masked stain in-distribution rather than a distribution shift.

8) Stain fusion: voting over single-stain models

Combines the per-stain MIL.py models from train_per_stain.sh without training anything: each
model's per-biopsy risk becomes a percentile of its own training-split scores (rank voting), and a
biopsy's fused risk is the mean over the stains it has (missing stains do not vote). With
--weighting cindex each stain votes with weight max(C_val - 0.5, 0), fixed on the validation split.

# Score every stain model on train/val/test, fuse (unweighted and C-index-weighted), and compare
# each fusion with each single stain on the test split (results/compare/voting_*_vs_*_test.txt)
./vote_across_stains.sh

# Or fuse existing evaluate_survival.py outputs directly
python python_scripts/evaluate_MIL/fuse_votes.py --models HE_HES PAS PASM Congo Trichrome Sirius_red Immune --pattern "results/eval/{split}/{model}/risk_scores.csv" --method rank --weighting cindex --output_dir results/eval/voting_rank_cindex

9) Stain fusion: late fusion (stacked Cox) over single-stain models

Learned counterpart of voting: a ridge-penalised Cox model whose features are each stain model's
normalised risk (0 when the stain is absent) plus one stain-presence indicator per stain. It is
fitted on the validation split — the single-stain models over-fit their training biopsies, so
their training scores would make it over-trust every stain — with the ridge strength picked by
5-fold cross-validation inside validation (one-standard-error rule). The test split is only scored.
A presence-only model is reported alongside, to show how much of the C-index comes from which stains
were ordered rather than from the images. Run as part of ./vote_across_stains.sh, or directly:

python python_scripts/evaluate_MIL/fuse_late.py --models HE_HES PAS PASM Congo Trichrome Sirius_red Immune --pattern "results/eval/{split}/{model}/risk_scores.csv" --output_dir results/eval/late_fusion
# Writes test/risk_scores.csv (for compare_cindex.py), coefficients.csv, cv.csv and summary.csv.

10) Stain fusion: early fusion (one bag of all stains' patches)

multistain_MIL.py --fusion early: every patch of every stain goes through one shared projection, a
learned stain embedding is added, and a single attention pool runs over the patches of all stains
at once, so patches of different stains compete directly (no stain-specific encoders, no per-stain
token stage as in UNICORN). --early_layers N adds N transformer layers over the joint bag first
(cost grows with the square of the total patch count). Everything else is shared with UNICORN:
--per_slide, --stain_dropout, --event_sampling, --resume, the val C-index / best checkpoint, the
test split, and analyze_stain_weights.py (per-stain contributions and ablation).

python python_scripts/MIL/multistain_MIL.py --task survival --fusion early --features_path WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2 --labels_path WSI/whole_registry/labels --labels_csv label_csvs/labels_super_stain.csv --stains HE-HES Congo Trichrome PAS PASM Immune --patches_per_stain 512 --stain_dropout 0.2 --event_sampling --checkpoint_dir checkpoints_early --log_dir results/losses_early

11) What the models capture: clinical baselines, within-diagnosis C-index, discordant pairs

analyze_model_signal.py compares any set of models (single stains, voting, late fusion, UNICORN,
early fusion — one --model NAME=path/{split}/risk_scores.csv each) on the test biopsies all of them
scored, and asks what their shared signal is: C-index with bootstrap CIs; kidney failure vs death as
endpoint; clinical Cox baselines (diagnosis, eGFR at biopsy, both, from the registry CSV) and each
model's added value over diagnosis+eGFR (C-index gain with paired bootstrap CI, likelihood-ratio
test); C-index within diagnosis; correlation of the models' risks (raw and after diagnosis+eGFR);
and discordant pairs (by time gap, overlap between models, biopsies most often ranked wrongly).

python python_scripts/evaluate_MIL/analyze_model_signal.py --model PAS="results/eval/{split}/PAS/risk_scores.csv" --model HE="results/eval/{split}/HE_HES/risk_scores.csv" --model voting="results/eval/voting_rank_none/{split}/risk_scores.csv" --model late="results/eval/late_fusion/{split}/risk_scores.csv" --output_dir results/model_signal

# The added-value Cox models are fitted on val: for a model itself fitted on val (late fusion) the
# likelihood-ratio p-value is in-sample — read the test-set C-index gain and its CI instead. The
# hazard ratio per SD is shared with diagnosis/eGFR when the risk is collinear with them.

12) n-year event probabilities and comparison at fixed horizons

survival_probability.py turns each model's risk score into P(event by t years): a Cox model on the
standardised risk plus a Breslow baseline hazard, both fitted on val, applied to test. It does the
same for diagnosis + eGFR and for diagnosis + eGFR + each model, and compares them at --horizons
(default 1 3 5 10 years) with time-dependent AUC and ROC curves per horizon, Brier score and IPA (1 - Brier / Brier of
Kaplan-Meier), calibration by decile and the test calibration slope, the integrated Brier score
and a decision curve, with bootstrap CIs and paired differences against --reference (default the
first model) and the clinical model. Same --model inputs as analyze_model_signal.py.

python python_scripts/evaluate_MIL/survival_probability.py --model PAS="results/eval/{split}/PAS/risk_scores.csv" --model voting="results/eval/voting_rank_none/{split}/risk_scores.csv" --model late="results/eval/late_fusion/{split}/risk_scores.csv" --model unicorn="results/eval/{split}/unicorn/risk_scores.csv" --model early="results/eval/{split}/early/risk_scores.csv" --output_dir results/survival_probability

# --endpoint kidney: kidney failure, with death as a competing risk (cause-specific Cox models,
# cumulative incidence; observed risks by Aalen-Johansen). Late fusion is recalibrated on the split
# it was fitted on, so its probabilities are slightly optimistic.
