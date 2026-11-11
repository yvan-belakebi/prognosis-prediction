#!/usr/bin/env bash
# All-stains model vs each per-stain model (both from train_per_stain.sh), stain
# by stain. For each stain:
#   - biopsies: those of SPLIT with at least one slide of that stain;
#   - both models score only that stain's slides, averaged per biopsy;
#   - paired C-index comparison on those biopsies (bootstrap + jackknife).
# Default split is test: val picked the best epoch of both models.
# Usage: ./compare_5_stains_vs_per_stain.sh   or   SPLIT=val ./compare_5_stains_vs_per_stain.sh
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV, STAINS and stain_tag — same as training.
source stain_config.sh

SPLIT=${SPLIT:-test}
# hidden_dim is read from each checkpoint by evaluate_survival.py.
MODEL_ARGS=(--model_type deepgraphsurv)
SHARED_CKPT=checkpoints/5_stains/deepgraphsurv_model_best.pth
if [[ ! -f "$SHARED_CKPT" ]]; then
    echo "No $SHARED_CKPT; train it with train_per_stain.sh first."
    exit 1
fi

score() {  # checkpoint output_dir stain
    local ckpt=$1 out=$2 stain=$3
    # Reuse scores only when newer than the checkpoint that made them.
    if [[ "$out/risk_scores.csv" -nt "$ckpt" ]]; then
        echo "    $out: up to date, reused"
        return 0
    fi
    mkdir -p "$out"
    python python_scripts/evaluate_MIL/evaluate_survival.py "${MODEL_ARGS[@]}" \
        --checkpoint "$ckpt" \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "$stain" --split "$SPLIT" --output_dir "$out" \
        > "$out/eval.log" 2>&1 \
        || { echo "!!! $out failed, see $out/eval.log"; return 1; }
}

mkdir -p results/compare
summary=results/compare/5_stains_vs_per_stain_$SPLIT.tsv
printf "stain\tn_biopsies\tevents\tc_5_stains\tc_per_stain\tdiff\tp_bootstrap\tp_jackknife\n" > "$summary"

for stain in "${STAINS[@]}"; do
    tag=$(stain_tag "$stain")
    ckpt="checkpoints/$tag/deepgraphsurv_model_best.pth"
    if [[ ! -f "$ckpt" ]]; then
        echo "--- $stain: no $ckpt, skipped"
        continue
    fi
    echo "=== $stain"
    own="results/eval/$SPLIT/$tag"
    shared="results/eval/$SPLIT/shared_$tag"
    score "$ckpt" "$own" "$stain" && score "$SHARED_CKPT" "$shared" "$stain" || continue

    # A = all-stains model, B = per-stain model; same stain, split and biopsies.
    out="results/compare/5_stains_vs_${tag}_$SPLIT.txt"
    python python_scripts/evaluate_MIL/compare_cindex.py \
        --inputs_a "$shared/risk_scores.csv" --inputs_b "$own/risk_scores.csv" \
        > "$out" 2>&1 || { echo "!!! comparison failed, see $out"; continue; }
    grep -E "^N=|C-index A|C-index B|Two-tailed" "$out"

    # One summary row: N, events, C_A, C_B, A-B, bootstrap p, jackknife p.
    awk -v s="$tag" '
        /^N=/              { split($0, a, /[=, ]+/); n = a[2]; ev = a[4] }
        /C-index A:/       { ca = $NF }
        /C-index B:/       { cb = $NF }
        /Observed diff/    { d = $NF }
        /Two-tailed/       { p[++k] = $3 }
        END { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", s, n, ev, ca, cb, d, p[1], (k > 1 ? p[2] : "NA") }
    ' "$out" >> "$summary"
done

echo
echo "=== summary (A = 5_stains, B = per-stain) -> $summary"
column -t -s $'\t' "$summary" 2>/dev/null || cat "$summary"
