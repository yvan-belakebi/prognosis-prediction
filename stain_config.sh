# Shared settings for train_per_stain.sh and compare_accross_stains.sh.
# Sourced, not run: `source stain_config.sh`.
FEATURES=WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2
LABELS=WSI/whole_registry/labels
LABELS_CSV=label_csvs/labels_super_stain.csv

# One entry per model; '|' merges stain names into one model stain.
# Names must match the stain column of LABELS_CSV: in the super-stain CSV the
# immunostains (IgA, IgG, IgM, C3, C1q, C5-9, Kappa, Lambda) are all "Immune".
STAINS=("HE-HES" "PAS" "PASM" "Congo" "Trichrome" "Sirius red" "C1q" "C3" "C5-9" "IgA" "IgG" "IgM" "Kappa" "Lambda")

# Folder-safe name of a stain: "HE|HES" -> "HE_HES", "Sirius red" -> "Sirius_red"
stain_tag() { echo "$1" | tr -c 'A-Za-z0-9\n' '_'; }
