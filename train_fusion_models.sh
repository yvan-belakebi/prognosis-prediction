#!/usr/bin/env bash
# Train the multi-stain fusion models (multistain_MIL.py), each in its own
# folders, then score them on val and test. The GPUs of GPUS (default "0 1") run
# side by side, one variant each: the 1st variant on GPU 0, the 2nd on GPU 1,
# and each further one on the first GPU to become free:
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
#        GPUS=0 ./train_fusion_models.sh                (all on GPU 0, one at a time)
# The runs on different GPUs share the CPU cores and /dev/shm: multistain_MIL.py
# lowers --num_workers to its share of /dev/shm (--shm_share).
# A variant with a checkpoints_<name>/multistain_survival_last.pth is resumed
# from it; one whose scores already exist is not rescored (delete them to redo).
set -u
cd "$(dirname "$0")"
# FEATURES, LABELS, LABELS_CSV and STAINS — same as the single-stain models.
source stain_config.sh

FUSIONS=(${FUSIONS:-unicorn early})
GPUS=(${GPUS:-0 1})

declare -A VARIANT_ARGS=(
    [unicorn]="--fusion unicorn"
    [unicorn_per_slide]="--fusion unicorn --per_slide"
    [early]="--fusion early"
    [early_tf]="--fusion early --early_layers 1"
)
# Shared by every variant, so they differ only in the fusion architecture.
COMMON_ARGS=(--task survival --patches_per_stain 512 --stain_dropout 0.2
             --event_sampling --batch_size 32 --epochs 50 --save_every 5 --num_workers 14
             --shm_share ${#GPUS[@]})
# The fusion models need no patch coordinates, so COORDS is not passed.
DATA_ARGS=(--features_path "$FEATURES" --file_ext "$FILE_EXT"
           --labels_path "$LABELS" --labels_csv "$LABELS_CSV")

# Train and score one variant on the GPU(s) in CUDA_VISIBLE_DEVICES. Runs in a
# background queue, so success is reported by a marker file in $done_dir.
run_variant() {
    local name=$1 gpu=$CUDA_VISIBLE_DEVICES
    local ckpt_dir="checkpoints_$name"
    local best="$ckpt_dir/multistain_survival_best.pth"
    local last="$ckpt_dir/multistain_survival_last.pth"
    local resume=()
    [[ -f "$last" ]] && resume=(--resume "$last")

    echo "=== [GPU $gpu] training $name -> $ckpt_dir${resume:+ (resuming)}"
    # shellcheck disable=SC2086  # VARIANT_ARGS holds several words on purpose
    python python_scripts/MIL/multistain_MIL.py "${COMMON_ARGS[@]}" "${DATA_ARGS[@]}" \
        ${VARIANT_ARGS[$name]} --stains "${STAINS[@]}" "${resume[@]}" \
        --checkpoint_dir "$ckpt_dir" --log_dir "results/losses_$name" \
        > "logs_train_$name.txt" 2>&1 \
        || { echo "!!! $name training failed, see logs_train_$name.txt"; return; }
    if [[ ! -f "$best" ]]; then
        echo "!!! $name: no $best, not scored"
        return
    fi

    local ok=1 split out skip
    for split in val test; do
        out="results/eval/$split/$name"
        if [[ -f "$out/risk_scores.csv" ]]; then
            echo "=== $out: already scored, reused"
            continue
        fi
        echo "=== [GPU $gpu] scoring $out"
        mkdir -p "$out"
        # The ablation (one pass per stain) is only needed on test.
        skip=()
        [[ $split == val ]] && skip=(--skip_ablation)
        python python_scripts/evaluate_MIL/analyze_stain_weights.py \
            --checkpoint "$best" "${DATA_ARGS[@]}" --split "$split" "${skip[@]}" \
            --output_dir "$out" > "$out/eval.log" 2>&1 \
            || { echo "!!! $out failed, see $out/eval.log"; ok=0; }
    done
    [[ $ok == 1 ]] && touch "$done_dir/$name"
}

names=()
for name in "${FUSIONS[@]}"; do
    if [[ -z ${VARIANT_ARGS[$name]+x} ]]; then
        echo "!!! unknown variant '$name' (one of: ${!VARIANT_ARGS[*]})"
        continue
    fi
    names+=("$name")
done

done_dir=$(mktemp -d)
trap 'rm -rf "$done_dir"' EXIT
# Each variant goes to the first GPU without a running job (gpu_pid holds the
# job of each GPU), waiting for any job to end when all GPUs are busy.
declare -A gpu_pid=()
for name in "${names[@]}"; do
    while :; do
        free=""
        for g in "${GPUS[@]}"; do
            pid=${gpu_pid[$g]:-}
            if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then free=$g; break; fi
        done
        [[ -n $free ]] && break
        wait -n  # a job ended (and is reaped, so kill -0 now fails for it)
    done
    CUDA_VISIBLE_DEVICES=$free run_variant "$name" &
    gpu_pid[$free]=$!
done
wait

scored=()
for name in "${names[@]}"; do
    [[ -f "$done_dir/$name" ]] && scored+=("$name")
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
