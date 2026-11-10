#!/usr/bin/env bash
# Train the multi-stain fusion models (multistain_MIL.py), one after the other,
# each in its own folders, then score them on val and test:
#   1. train every variant in FUSIONS on the stains of stain_config.sh;
#   2. score val and test with analyze_stain_weights.py (risk_scores.csv, plus
#      the stain contributions and, on test, the leave-one-stain-out ablation);
#   3. compare each variant on test with PAS and with the stain fusions of
#      vote_across_stains.sh, when those scores exist.
#
# Variants (name: multistain_MIL.py arguments):
#   unicorn            per-stain encoders -> one token per stain -> transformer
#   unicorn_per_slide  same, each slide encoded separately, slides pooled by attention
#   early              one shared projection + stain embedding, one attention pool
#                      over the patches of all stains
#   early_tf           early, with one transformer layer over the joint bag first
#
# Usage: ./train_fusion_models.sh                       (unicorn and early)
#        FUSIONS="unicorn early early_tf" ./train_fusion_models.sh
# A variant with a checkpoints_<name>/multistain_survival_last.pth is resumed
# from it; one whose scores already exist is not rescored (delete them to redo).
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV and STAINS — same as the single-stain models.
source stain_config.sh

FUSIONS=(${FUSIONS:-unicorn early})

declare -A VARIANT_ARGS=(
    [unicorn]="--fusion unicorn"
    [unicorn_per_slide]="--fusion unicorn --per_slide"
    [early]="--fusion early"
    [early_tf]="--fusion early --early_layers 1"
)
# Shared by every variant, so they differ only in the fusion architecture.
COMMON_ARGS=(--task survival --patches_per_stain 512 --stain_dropout 0.2
             --event_sampling --batch_size 8 --epochs 50 --save_every 5 --num_workers 14)
DATA_ARGS=(--features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV")

scored=()
for name in "${FUSIONS[@]}"; do
    if [[ -z ${VARIANT_ARGS[$name]+x} ]]; then
        echo "!!! unknown variant '$name' (one of: ${!VARIANT_ARGS[*]})"
        continue
    fi
    ckpt_dir="checkpoints_$name"
    best="$ckpt_dir/multistain_survival_best.pth"
    last="$ckpt_dir/multistain_survival_last.pth"
    resume=()
    [[ -f "$last" ]] && resume=(--resume "$last")

    echo "=== training $name -> $ckpt_dir${resume:+ (resuming)}"
    # shellcheck disable=SC2086  # VARIANT_ARGS holds several words on purpose
    python python_scripts/MIL/multistain_MIL.py "${COMMON_ARGS[@]}" "${DATA_ARGS[@]}" \
        ${VARIANT_ARGS[$name]} --stains "${STAINS[@]}" "${resume[@]}" \
        --checkpoint_dir "$ckpt_dir" --log_dir "results/losses_$name" \
        > "logs_train_$name.txt" 2>&1 \
        || { echo "!!! $name training failed, see logs_train_$name.txt"; continue; }
    if [[ ! -f "$best" ]]; then
        echo "!!! $name: no $best, not scored"
        continue
    fi

    ok=1
    for split in val test; do
        out="results/eval/$split/$name"
        if [[ -f "$out/risk_scores.csv" ]]; then
            echo "=== $out: already scored, reused"
            continue
        fi
        echo "=== scoring $out"
        mkdir -p "$out"
        # The ablation (one pass per stain) is only needed on test.
        skip=()
        [[ $split == val ]] && skip=(--skip_ablation)
        python python_scripts/evaluate_MIL/analyze_stain_weights.py \
            --checkpoint "$best" "${DATA_ARGS[@]}" --split "$split" "${skip[@]}" \
            --output_dir "$out" > "$out/eval.log" 2>&1 \
            || { echo "!!! $out failed, see $out/eval.log"; ok=0; }
    done
    [[ $ok == 1 ]] && scored+=("$name")
done

# Paired test-set comparisons, on the biopsies both scored.
mkdir -p results/compare
compare() {  # name_a csv_a name_b csv_b
    local out="results/compare/${1}_vs_${3}_test.txt"
    if [[ ! -f "$2" || ! -f "$4" ]]; then
        echo "--- $1 vs $3: missing scores, skipped"
        return
    fi
    python python_scripts/evaluate_MIL/compare_cindex.py \
        --inputs_a "$2" --inputs_b "$4" > "$out" 2>&1 \
        || echo "!!! comparison failed, see $out"
    echo "--- $1 vs $3:"
    grep -E "^N=|C-index A|C-index B|Two-tailed" "$out"
}
for name in "${scored[@]}"; do
    csv="results/eval/test/$name/risk_scores.csv"
    compare "$name" "$csv" PAS results/eval/test/PAS/risk_scores.csv
    compare "$name" "$csv" voting_none results/eval/voting_rank_none/test/risk_scores.csv
    compare "$name" "$csv" late_fusion results/eval/late_fusion/test/risk_scores.csv
done
for ((i = 0; i < ${#scored[@]}; i++)); do
    for ((j = i + 1; j < ${#scored[@]}; j++)); do
        a=${scored[i]} b=${scored[j]}
        compare "$a" "results/eval/test/$a/risk_scores.csv" "$b" "results/eval/test/$b/risk_scores.csv"
    done
done
