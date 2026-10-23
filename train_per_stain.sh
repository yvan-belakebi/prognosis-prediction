#!/usr/bin/env bash
# Train one DeepGraphSurv model per stain, sequentially, each in its own folders.
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV, STAINS and stain_tag, shared with
# compare_accross_stains.sh so evaluation always matches training.
source stain_config.sh

for stain in "${STAINS[@]}"; do
    tag=$(stain_tag "$stain")
    echo "=== $stain -> checkpoints/$tag ==="
    python python_scripts/MIL/MIL.py \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "$stain" \
        --model_type deepgraphsurv --hidden_dim 128 --dropout 0.25 --epochs 50 \
        --lr 3e-5 --weight_decay 1e-3 --max_patches 4096 \
        --batch_size 32 --biopsy_sampling --num_workers 14 --save_every 5 \
        --checkpoint_dir "checkpoints/$tag" --log_dir "results/losses/$tag" \
        > "logs_train_$tag.txt" 2>&1 \
        || echo "!!! $stain failed, see logs_train_$tag.txt"
done

# One model on the slides of every stain in STAINS (same architecture as above,
# so it compares directly with the per-stain models). Several times more data
# per epoch, hence a lower learning rate. With --biopsy_sampling each epoch uses
# one slide per biopsy, of whichever listed stain is drawn.
tag=all_stains
echo "=== all stains (${STAINS[*]}) -> checkpoints/$tag ==="
python python_scripts/MIL/MIL.py \
    --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
    --stains "${STAINS[@]}" \
    --model_type deepgraphsurv --hidden_dim 128 --dropout 0.25 --epochs 50 \
    --lr 1e-5 --weight_decay 1e-3 --max_patches 4096 \
    --batch_size 32 --num_workers 14 --save_every 5 \
    --checkpoint_dir "checkpoints/$tag" --log_dir "results/losses/$tag" \
    > "logs_train_$tag.txt" 2>&1 \
    || echo "!!! all-stains model failed, see logs_train_$tag.txt"
