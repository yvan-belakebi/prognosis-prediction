#!/usr/bin/env bash
# Voting and late fusion of per-stain risk scores:
#   1. score every stain on the train, val and test splits
#      (train is the normalisation reference, val sets the optional weights);
#   2. fuse the scores by rank voting, unweighted and C-index-weighted;
#   3. fit late fusion (stacked Cox) on val;
#   4. compare every fusion with each single stain, and with each other, on test.
#
# Which model scores each stain (both from train_per_stain.sh):
#   SCORER=per_stain  (default) each stain by its own model, checkpoints/<stain>/
#                     -> results/eval/{split}/<stain>, voting_rank_*, late_fusion
#   SCORER=all_stains every stain by the one model trained on all stains,
#                     checkpoints/all_stains/
#                     -> results/eval/{split}/shared_<stain>, voting_rank_*_shared,
#                        late_fusion_shared; the fusions are also compared with that
#                        model averaged over all of a biopsy's slides (all_stains_mean)
#                        and with the per-stain fusions when those exist.
# Usage: ./vote_across_stains.sh   or   SCORER=all_stains ./vote_across_stains.sh
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV, STAINS and stain_tag — same as training.
source stain_config.sh

SCORER=${SCORER:-per_stain}
case "$SCORER" in
    per_stain)  prefix="";        suffix="" ;;
    all_stains) prefix="shared_"; suffix="_shared" ;;
    *) echo "SCORER must be per_stain or all_stains, not '$SCORER'"; exit 1 ;;
esac

# hidden_dim is read from each checkpoint by evaluate_survival.py.
MODEL_ARGS=(--model_type deepgraphsurv)

score() {  # checkpoint split output_dir stain...
    local ckpt=$1 split=$2 out=$3
    shift 3
    if [[ -f "$out/risk_scores.csv" ]]; then
        echo "=== $out: already scored, reused"
        return 0
    fi
    echo "=== scoring $out"
    mkdir -p "$out"
    python python_scripts/evaluate_MIL/evaluate_survival.py "${MODEL_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "$@" --split "$split" --output_dir "$out" \
        > "$out/eval.log" 2>&1 \
        || { echo "!!! $out failed, see $out/eval.log"; return 1; }
}

models=()
for stain in "${STAINS[@]}"; do
    tag=$(stain_tag "$stain")
    if [[ $SCORER == all_stains ]]; then
        ckpt="checkpoints/all_stains/deepgraphsurv_model_best.pth"
    else
        ckpt="checkpoints/$tag/deepgraphsurv_model_best.pth"
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "--- $stain: no $ckpt, left out of the vote"
        continue
    fi
    name="$prefix$tag"
    ok=1
    for split in train val test; do
        score "$ckpt" "$split" "results/eval/$split/$name" "$stain" || ok=0
    done
    [[ $ok == 1 ]] && models+=("$name")
done
if [[ ${#models[@]} -eq 0 ]]; then
    echo "No stain was scored; nothing to fuse."
    exit 1
fi

for weighting in none cindex; do
    echo "=== rank voting, weighting=$weighting"
    python python_scripts/evaluate_MIL/fuse_votes.py \
        --models "${models[@]}" \
        --pattern "results/eval/{split}/{model}/risk_scores.csv" \
        --method rank --weighting "$weighting" \
        --output_dir "results/eval/voting_rank_$weighting$suffix"
done

# Late fusion: a ridge Cox model on the same normalised scores (+ stain-presence
# indicators), fitted on val with cross-validated ridge strength, scored on test.
echo "=== late fusion (stacked Cox, fitted on val)"
python python_scripts/evaluate_MIL/fuse_late.py \
    --models "${models[@]}" \
    --pattern "results/eval/{split}/{model}/risk_scores.csv" \
    --method rank --output_dir "results/eval/late_fusion$suffix"

# Paired test-set comparisons, on the biopsies both scored: each fusion against
# each single stain, and late fusion against both votes.
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
vote() { echo "results/eval/voting_rank_$1$2/test/risk_scores.csv"; }  # weighting suffix
late=results/eval/late_fusion$suffix/test/risk_scores.csv
for name in "${models[@]}"; do
    single="results/eval/test/$name/risk_scores.csv"
    for weighting in none cindex; do
        compare "voting_$weighting$suffix" "$(vote $weighting "$suffix")" "$name" "$single"
    done
    compare "late_fusion$suffix" "$late" "$name" "$single"
done
for weighting in none cindex; do
    compare "late_fusion$suffix" "$late" "voting_$weighting$suffix" "$(vote $weighting "$suffix")"
done

if [[ $SCORER == all_stains ]]; then
    # The all-stains model without fusion: the mean of its scores over every slide of
    # a biopsy, whatever the stain. Fusion has to beat this to be worth it.
    mean=results/eval/test/all_stains_mean
    score "checkpoints/all_stains/deepgraphsurv_model_best.pth" test "$mean" "${STAINS[@]}"
    for weighting in none cindex; do
        compare "voting_$weighting$suffix" "$(vote $weighting "$suffix")" \
            all_stains_mean "$mean/risk_scores.csv"
    done
    compare "late_fusion$suffix" "$late" all_stains_mean "$mean/risk_scores.csv"
    # Shared model vs per-stain models, fusion for fusion (run SCORER=per_stain first).
    for weighting in none cindex; do
        compare "voting_$weighting$suffix" "$(vote $weighting "$suffix")" \
            "voting_$weighting" "$(vote $weighting "")"
    done
    compare "late_fusion$suffix" "$late" late_fusion results/eval/late_fusion/test/risk_scores.csv
fi
