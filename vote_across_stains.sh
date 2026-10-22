#!/usr/bin/env bash
# Voting and late fusion of the per-stain models from train_per_stain.sh:
#   1. score every stain model on the train, val and test splits
#      (train is the normalisation reference, val sets the optional weights);
#   2. fuse the scores by rank voting, unweighted and C-index-weighted;
#   3. fit late fusion (stacked Cox) on val;
#   4. compare every fusion with each single stain, and with each other, on test.
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV, STAINS and stain_tag — same as training.
source stain_config.sh

# Must match the architecture arguments used in train_per_stain.sh.
MODEL_ARGS=(--model_type deepgraphsurv --hidden_dim 128)

models=()
for stain in "${STAINS[@]}"; do
    tag=$(stain_tag "$stain")
    ckpt="checkpoints/$tag/deepgraphsurv_model_best.pth"
    if [[ ! -f "$ckpt" ]]; then
        echo "--- $stain: no $ckpt, left out of the vote"
        continue
    fi
    ok=1
    for split in train val test; do
        out="results/eval/$split/$tag"
        if [[ -f "$out/risk_scores.csv" ]]; then
            echo "=== $stain / $split: already scored ($out), reused"
            continue
        fi
        echo "=== scoring $stain on $split"
        mkdir -p "$out"
        python python_scripts/evaluate_MIL/evaluate_survival.py "${MODEL_ARGS[@]}" \
            --checkpoint "$ckpt" \
            --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
            --stains "$stain" --split "$split" --output_dir "$out" \
            > "$out/eval.log" 2>&1 \
            || { echo "!!! $stain on $split failed, see $out/eval.log"; ok=0; }
    done
    [[ $ok == 1 ]] && models+=("$tag")
done

for weighting in none cindex; do
    echo "=== rank voting, weighting=$weighting"
    python python_scripts/evaluate_MIL/fuse_votes.py \
        --models "${models[@]}" \
        --pattern "results/eval/{split}/{model}/risk_scores.csv" \
        --method rank --weighting "$weighting" \
        --output_dir "results/eval/voting_rank_$weighting"
done

# Late fusion: a ridge Cox model on the same normalised scores (+ stain-presence
# indicators), fitted on val with cross-validated ridge strength, scored on test.
echo "=== late fusion (stacked Cox, fitted on val)"
python python_scripts/evaluate_MIL/fuse_late.py \
    --models "${models[@]}" \
    --pattern "results/eval/{split}/{model}/risk_scores.csv" \
    --method rank --output_dir results/eval/late_fusion

# Paired test-set comparisons, on the biopsies both scored: each fusion against
# each single stain, and late fusion against both votes.
mkdir -p results/compare
compare() {  # name_a csv_a name_b csv_b
    local out="results/compare/${1}_vs_${3}_test.txt"
    python python_scripts/evaluate_MIL/compare_cindex.py \
        --inputs_a "$2" --inputs_b "$4" > "$out" 2>&1 \
        || echo "!!! comparison failed, see $out"
    echo "--- $1 vs $3:"
    grep -E "^N=|C-index A|C-index B|Two-tailed" "$out"
}
late=results/eval/late_fusion/test/risk_scores.csv
for tag in "${models[@]}"; do
    single="results/eval/test/$tag/risk_scores.csv"
    for weighting in none cindex; do
        compare "voting_$weighting" "results/eval/voting_rank_$weighting/test/risk_scores.csv" \
            "$tag" "$single"
    done
    compare late_fusion "$late" "$tag" "$single"
done
for weighting in none cindex; do
    compare late_fusion "$late" "voting_$weighting" \
        "results/eval/voting_rank_$weighting/test/risk_scores.csv"
done
