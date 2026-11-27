#!/usr/bin/env bash
# Train the multi-stain fusion models (multistain_MIL.py), each in its own
# folders, then score them on val and test. Every (foundation model, variant,
# seed) is one run; the GPUs of GPUS (default "0 1") run side by side, one run
# each: the 1st run on GPU 0, the 2nd on GPU 1, and each further one on the
# first GPU to become free:
#   1. train every variant in FUSIONS, with every seed in SEEDS, on the features
#      of every foundation model in FMS, on the stains of stain_config.sh;
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
# Each run is named <variant>_s<seed> and filed under its foundation model:
#   checkpoints_<FM>/<variant>_s<seed>/            checkpoints + multistain_config.json
#   results/eval/<split>/<FM>/<variant>_s<seed>/   risk_scores.csv, ...
#   results/losses_<FM>/<variant>_s<seed>/         loss log
#   logs_train_<FM>_<variant>_s<seed>.txt
# so runs on different features, or with different seeds, never resume from or
# reuse each other's outputs.
#
# Foundation models: FMS lists them (default: FM of stain_config.sh, uni_v2).
# Each one's features are in fm_features <FM> (stain_config.sh:
# .../features_<FM>_npy) unless FEATURES_<FM> names the folder; with a single FM,
# FM and FEATURES work as well. The queue runs seed by seed, so the seed-0 runs of
# every FM finish first.
#
# Usage: ./train_fusion_models.sh                       (unicorn and early, uni_v2, seed 0)
#        FUSIONS="unicorn early early_tf" ./train_fusion_models.sh
#        FM=virchow2 ./train_fusion_models.sh            (features_virchow2_npy)
#        FM=virchow2 FEATURES=path/to/feats ./train_fusion_models.sh
#        SEEDS="0 1 2" ./train_fusion_models.sh          (3 runs per variant)
#        FMS="uni_v2 virchow2" SEEDS="0 1 2" ./train_fusion_models.sh   (12 runs, 2 GPUs)
#        FMS="uni_v2 conch_v15" FEATURES_conch_v15=path/to/feats ./train_fusion_models.sh
#        GPUS=0 ./train_fusion_models.sh                (all on GPU 0, one at a time)
# The runs on different GPUs share the CPU cores and /dev/shm: multistain_MIL.py
# lowers --num_workers to its share of /dev/shm, RAM and cores (--shm_share).
# A run with a multistain_survival_last.pth in its checkpoint folder is resumed
# from it; one whose scores already exist is not rescored (delete them to redo).
set -u
cd "$(dirname "$0")"
# FM, FEATURES, fm_features, LABELS, LABELS_CSV and STAINS — same as the
# single-stain models.
source stain_config.sh

FUSIONS=(${FUSIONS:-unicorn early})
SEEDS=(${SEEDS:-0})
GPUS=(${GPUS:-0 1})
declare -A FEAT=()  # feature folder per foundation model
if [[ -z ${FMS:-} ]]; then
    FMS=("$FM")
    FEAT[$FM]=$FEATURES
else
    FMS=($FMS)
    for fm in "${FMS[@]}"; do
        var="FEATURES_$fm"
        FEAT[$fm]=${!var:-$(fm_features "$fm")}
    done
fi
for fm in "${FMS[@]}"; do
    if [[ ! -d "${FEAT[$fm]}" ]]; then
        echo "No feature folder ${FEAT[$fm]} for $fm; set FEATURES_$fm."
        exit 1
    fi
    echo "=== foundation model $fm: ${FEAT[$fm]}"
done

declare -A VARIANT_ARGS=(
    [unicorn]="--fusion unicorn"
    [unicorn_per_slide]="--fusion unicorn --per_slide"
    [early]="--fusion early"
    [early_tf]="--fusion early --early_layers 1"
)
# Shared by every variant, so they differ only in the fusion architecture.
COMMON_ARGS=(--task survival --patches_per_stain 256 --stain_dropout 0.2
             --event_sampling --batch_size 32 --epochs 50 --save_every 5 --num_workers 6
             --shm_share ${#GPUS[@]})
# Train and score one variant with one seed on one foundation model's features,
# on the GPU(s) in CUDA_VISIBLE_DEVICES. Runs in a background queue, so success
# is reported by a marker file in $done_dir.
run_variant() {
    local FM=$1 name=$2 seed=$3 gpu=$CUDA_VISIBLE_DEVICES
    local run="${name}_s${seed}"
    local ckpt_dir="checkpoints_$FM/$run"
    # The fusion models need no patch coordinates, so COORDS is not passed.
    local DATA_ARGS=(--features_path "${FEAT[$FM]}" --file_ext "$FILE_EXT"
                     --labels_path "$LABELS" --labels_csv "$LABELS_CSV")
    local best="$ckpt_dir/multistain_survival_best.pth"
    local last="$ckpt_dir/multistain_survival_last.pth"
    local resume=()
    [[ -f "$last" ]] && resume=(--resume "$last")

    local log="logs_train_${FM}_$run.txt"
    echo "=== [GPU $gpu] training $FM/$run -> $ckpt_dir${resume:+ (resuming)}"
    # shellcheck disable=SC2086  # VARIANT_ARGS holds several words on purpose
    python python_scripts/MIL/multistain_MIL.py "${COMMON_ARGS[@]}" "${DATA_ARGS[@]}" \
        ${VARIANT_ARGS[$name]} --stains "${STAINS[@]}" "${resume[@]}" --seed "$seed" \
        --checkpoint_dir "$ckpt_dir" --log_dir "results/losses_$FM/$run" \
        > "$log" 2>&1 \
        || { echo "!!! $FM/$run training failed, see $log"; return; }
    if [[ ! -f "$best" ]]; then
        echo "!!! $FM/$run: no $best, not scored"
        return
    fi

    local ok=1 split out skip elog
    for split in val test; do
        out="results/eval/$split/$FM/$run"
        # Scored before embeddings were saved: add them only (no rescoring).
        if [[ -f "$out/risk_scores.csv" ]]; then
            if [[ -f "$out/embeddings.npz" ]]; then
                echo "=== $out: already scored, reused"
                continue
            fi
            skip=(--embeddings_only) elog="$out/embeddings.log"
            echo "=== [GPU $gpu] embeddings for $out"
        else
            # The ablation (one pass per stain) is only needed on test.
            skip=() elog="$out/eval.log"
            [[ $split == val ]] && skip=(--skip_ablation)
            echo "=== [GPU $gpu] scoring $out"
        fi
        mkdir -p "$out"
        python python_scripts/evaluate_MIL/analyze_stain_weights.py \
            --checkpoint "$best" "${DATA_ARGS[@]}" --split "$split" "${skip[@]}" --save_embeddings \
            --output_dir "$out" > "$elog" 2>&1 \
            || { echo "!!! $out failed, see $elog"; ok=0; }
    done
    [[ $ok == 1 ]] && touch "$done_dir/${FM}__$run"
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
# Each run goes to the first GPU without a running job (gpu_pid holds the job
# of each GPU), waiting for any job to end when all GPUs are busy.
declare -A gpu_pid=()
for seed in "${SEEDS[@]}"; do
 for name in "${names[@]}"; do
  for fm in "${FMS[@]}"; do
    while :; do
        free=""
        for g in "${GPUS[@]}"; do
            pid=${gpu_pid[$g]:-}
            if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then free=$g; break; fi
        done
        [[ -n $free ]] && break
        wait -n  # a job ended (and is reaped, so kill -0 now fails for it)
    done
    CUDA_VISIBLE_DEVICES=$free run_variant "$fm" "$name" "$seed" &
    gpu_pid[$free]=$!
  done
 done
done
wait

scored=()  # FM/<variant>_s<seed>
for fm in "${FMS[@]}"; do
    for name in "${names[@]}"; do
        for seed in "${SEEDS[@]}"; do
            [[ -f "$done_dir/${fm}__${name}_s${seed}" ]] && scored+=("$fm/${name}_s${seed}")
        done
    done
done

# Paired test-set comparisons, on the biopsies both scored.
mkdir -p results/compare
compare() {  # name_a csv_a name_b csv_b
    local out="results/compare/${1//\//_}_vs_${3//\//_}_test.txt"
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
# PAS, voting and late fusion come from train_per_stain.sh / vote_across_stains.sh
# on uni_v2 features, so only uni_v2 runs are compared with them.
for run in "${scored[@]}"; do
    [[ ${run%%/*} == uni_v2 ]] || continue
    csv="results/eval/test/$run/risk_scores.csv"
    compare "$run" "$csv" PAS results/eval/test/PAS/risk_scores.csv
    compare "$run" "$csv" voting_none results/eval/voting_rank_none/test/risk_scores.csv
    compare "$run" "$csv" late_fusion results/eval/late_fusion/test/risk_scores.csv
done
# Every pair of runs on one FM; across FMs, only the same variant and seed
# (compare_foundation_models.py covers the rest).
for ((i = 0; i < ${#scored[@]}; i++)); do
    for ((j = i + 1; j < ${#scored[@]}; j++)); do
        a=${scored[i]} b=${scored[j]}
        [[ ${a%%/*} == "${b%%/*}" || ${a#*/} == "${b#*/}" ]] || continue
        compare "$a" "results/eval/test/$a/risk_scores.csv" \
                "$b" "results/eval/test/$b/risk_scores.csv"
    done
done
if (( $(printf '%s\n' "${scored[@]%%/*}" | sort -u | wc -l) > 1 )); then
    echo "=== Compare the foundation models:"
    echo "    python python_scripts/evaluate_MIL/compare_foundation_models.py" \
         "--fms ${FMS[*]} --variants ${names[*]}"
fi
