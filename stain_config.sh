# Shared settings for train_per_stain.sh and compare_accross_stains.sh.
# Sourced, not run: `source stain_config.sh`.
# Features: float16 .npy copies of TRIDENT's UNI-v2 .h5 features
# (python_scripts/MIL/h5_features_to_npy.py). The patch coordinates, needed by the
# graph models, stay in TRIDENT's patches/{slide}_patches.h5.
# FM names the foundation model (TRIDENT's encoder name, e.g. uni_v2, virchow2,
# conch_v15): it picks the feature folder, and train_fusion_models.sh files its
# checkpoints and scores under it. Set FEATURES too if the folder is named otherwise.
FM=${FM:-uni_v2}
# Feature folder of a foundation model, by TRIDENT's naming.
fm_features() { echo "WSI/whole_registry/20x_224px_0px_overlap/features_${1}_npy"; }
FEATURES=${FEATURES:-$(fm_features "$FM")}
FILE_EXT=.npy
COORDS=WSI/whole_registry/20x_224px_0px_overlap/patches
LABELS=WSI/whole_registry/labels
LABELS_CSV=label_csvs/labels.csv

# One entry per model; '|' merges stain names into one model stain, named after
# the first entry (which need not exist in the CSV): "HE_HES|HE|HES" trains one
# HE_HES model on the HE and HES slides. The stain column of LABELS_CSV keeps
# the original stain, so stain normalisation stays per original stain.
# The other names must match that column: in the super-stain CSV the
# immunostains (IgA, IgG, IgM, C3, C1q, C5-9, Kappa, Lambda) are all "Immune".
STAINS=("HE_HES|HE|HES" "PAS" "PASM" "Congo" "Trichrome|AFOG|Masson Trichrom" "Sirius red" "C1q" "C3" "C5-9" "IgA" "IgG" "IgM" "Kappa" "Lambda")

# Folder-safe name of a stain, from its first name:
# "HE_HES|HE|HES" -> "HE_HES", "Sirius red" -> "Sirius_red"
stain_tag() { echo "${1%%|*}" | tr -c 'A-Za-z0-9\n' '_'; }
