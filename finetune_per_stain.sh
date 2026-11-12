#!/usr/bin/env bash
# Fine-tune the 5_stains model (HE_HES, PAS, Congo, Trichrome; trained on biopsies
# without the stains below) into one model per stain below, sequentially.
# LR and WD can be overridden: LR=1e-5 WD=1e-3 ./finetune_per_stain.sh
set -u
cd "$(dirname "$0")"
source stain_config.sh   # FEATURES, LABELS, LABELS_CSV, stain_tag

PRETRAINED=checkpoints/5_stains/deepgraphsurv_model_best.pth
FT_STAINS=("Sirius red" "C1q" "C3" "C5-9" "IgA" "IgG" "IgM" "Kappa" "Lambda")
# Half the pretraining lr: start near the pretrained optimum without wiping it.
# 10x the pretraining weight decay: more regularisation for 1-10x fewer biopsies.
LR=${LR:-5e-6}
WD=${WD:-1e-2}

if [[ ! -f "$PRETRAINED" ]]; then
    echo "No $PRETRAINED; train the 5_stains model first."
    exit 1
fi

for stain in "${FT_STAINS[@]}"; do
    tag=ft_$(stain_tag "$stain")
    echo "=== $stain -> checkpoints/$tag"
    # Architecture (model_type, hidden_dim) and max_patches as in pretraining.
    python python_scripts/MIL/MIL.py \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "$stain" \
        --authorized_slides_csv label_csvs/slides_with_selected_stains.csv \
        --init_checkpoint "$PRETRAINED" \
        --model_type deepgraphsurv --hidden_dim 128 --dropout 0.25 --epochs 25 \
        --lr "$LR" --weight_decay "$WD" --max_patches 2048 \
        --batch_size 32 --biopsy_sampling --num_workers 14 --save_every 5 \
        --checkpoint_dir "checkpoints/$tag" --log_dir "results/losses/$tag" \
        > "logs_train_$tag.txt" 2>&1 \
        || echo "!!! $stain failed, see logs_train_$tag.txt"
done
