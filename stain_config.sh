# Shared settings for train_per_stain.sh and compare_accross_stains.sh.
# Sourced, not run: `source stain_config.sh`.
FEATURES=WSI/whole_registry/20x_224px_0px_overlap/features_uni_v2
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
