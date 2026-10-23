#!/usr/bin/env bash
# Evaluate each per-stain model (from train_per_stain.sh) on its validation
# biopsies, then compare every pair of stains on the biopsies both scored.
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV, STAINS and stain_tag — same as training.
source stain_config.sh

# Must match the architecture arguments used in train_per_stain.sh.
MODEL_ARGS=(--model_type deepgraphsurv --hidden_dim 64)

evaluated=()
for stain in "${STAINS[@]}"; do
    tag=$(stain_tag "$stain")
    ckpt="checkpoints/$tag/deepgraphsurv_model_best.pth"
    if [[ ! -f "$ckpt" ]]; then
        echo "--- $stain: no $ckpt, skipped"
        continue
    fi
    echo "=== evaluating $stain"
    if python python_scripts/evaluate_MIL/evaluate_survival.py "${MODEL_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "$stain" --output_dir "results/eval/$tag" \
        > "results/eval_$tag.log" 2>&1; then
        grep "C-index:" "results/eval_$tag.log"
        evaluated+=("$tag")
    else
        echo "!!! $stain evaluation failed, see results/eval_$tag.log"
    fi
done

# Paired comparison of every pair: same biopsies, bootstrap + jackknife tests.
mkdir -p results/compare
for ((i = 0; i < ${#evaluated[@]}; i++)); do
    for ((j = i + 1; j < ${#evaluated[@]}; j++)); do
        a=${evaluated[i]} b=${evaluated[j]}
        out="results/compare/${a}_vs_${b}.txt"
        echo "=== $a vs $b -> $out"
        python python_scripts/evaluate_MIL/compare_cindex.py \
            --inputs_a "results/eval/$a/risk_scores.csv" \
            --inputs_b "results/eval/$b/risk_scores.csv" > "$out" 2>&1 \
            || echo "!!! comparison failed, see $out"
        grep -E "^N=|C-index A|C-index B|Two-tailed" "$out"
    done
done
