#!/usr/bin/env bash
# run_multi_model.sh — Launch multi-model replication experiments in parallel.
#
# Runs experiments 1, 8, 11, 13 on three additional models (Llama-1B, Qwen-1.5B,
# Gemma-2B) with proper GPU assignment.  Existing Llama-3B results are symlinked.
#
# Usage:
#   bash scripts/run_multi_model.sh           # full run
#   bash scripts/run_multi_model.sh --dry-run # print commands only

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_RUN="conda run --no-capture-output -n icl"

# ---------------------------------------------------------------------------
# Models (excluding the existing Llama-3B which already has results)
# ---------------------------------------------------------------------------
declare -A MODELS
MODELS["llama-1b"]="meta-llama/Llama-3.2-1B-Instruct"
MODELS["qwen-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
MODELS["gemma-2b"]="google/gemma-2-2b-it"

# Short names used in result directories (must match model.short_name output)
declare -A SHORT_NAMES
SHORT_NAMES["llama-1b"]="llama-3.2-1b-instruct"
SHORT_NAMES["qwen-1.5b"]="qwen2.5-1.5b-instruct"
SHORT_NAMES["gemma-2b"]="gemma-2-2b-it"

# ---------------------------------------------------------------------------
# GPU assignments (adjust to your hardware)
# ---------------------------------------------------------------------------
declare -A GPU_EXP11  GPU_EXP8
GPU_EXP11["llama-1b"]=0
GPU_EXP11["qwen-1.5b"]=1
GPU_EXP11["gemma-2b"]=2
GPU_EXP8["llama-1b"]=3
GPU_EXP8["qwen-1.5b"]=4
GPU_EXP8["gemma-2b"]=5
GPU_SEQ=6   # GPU for sequential exp1 + exp13

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
LOG_DIR="results/runner_logs"
mkdir -p "$LOG_DIR"

# Create symlinks for existing Llama-3B results
LLAMA3B_SHORT="llama-3.2-3b-instruct"
for exp in exp1 exp8 exp11 exp13; do
    src="results/${exp}"
    dst="results/${LLAMA3B_SHORT}/${exp}"
    if [[ -d "$src" && ! -e "$dst" ]]; then
        mkdir -p "results/${LLAMA3B_SHORT}"
        ln -sfn "$(realpath "$src")" "$dst"
        echo "Symlinked $dst -> $src"
    fi
done

run_cmd() {
    local label="$1"
    shift
    local logfile="${LOG_DIR}/${label}.log"
    if $DRY_RUN; then
        echo "[DRY-RUN] $label: $*"
    else
        echo "[LAUNCH]  $label -> $logfile"
        $CONDA_RUN "$@" > "$logfile" 2>&1 &
    fi
}

PIDS=()

# ---------------------------------------------------------------------------
# Parallel block: exp11 (activation patching) — one per GPU
# ---------------------------------------------------------------------------
for key in llama-1b qwen-1.5b gemma-2b; do
    hf_id="${MODELS[$key]}"
    short="${SHORT_NAMES[$key]}"
    gpu="${GPU_EXP11[$key]}"
    outdir="results/${short}/exp11"

    run_cmd "exp11_${key}" \
        python scripts/11_activation_patching.py \
            --model "$hf_id" \
            --device "cuda:${gpu}" \
            --n-test 10 \
            --noise-scales 2.0 5.0 \
            --positions first_query_token \
            --output-dir "$outdir"
    if ! $DRY_RUN; then PIDS+=($!); fi
done

# ---------------------------------------------------------------------------
# Parallel block: exp8 (multi-position transplant) — one per GPU
# ---------------------------------------------------------------------------
for key in llama-1b qwen-1.5b gemma-2b; do
    hf_id="${MODELS[$key]}"
    short="${SHORT_NAMES[$key]}"
    gpu="${GPU_EXP8[$key]}"
    outdir="results/${short}/exp8"

    run_cmd "exp8_${key}" \
        python scripts/08_multi_position.py \
            --model "$hf_id" \
            --device "cuda:${gpu}" \
            --n-test 10 \
            --output-dir "$outdir"
    if ! $DRY_RUN; then PIDS+=($!); fi
done

# ---------------------------------------------------------------------------
# Sequential block on GPU_SEQ: exp1 (baseline) then exp13 (instance analysis)
# ---------------------------------------------------------------------------
seq_log="${LOG_DIR}/seq_exp1_exp13.log"
if $DRY_RUN; then
    echo "[DRY-RUN] sequential exp1+exp13 on cuda:${GPU_SEQ}"
else
    echo "[LAUNCH]  sequential exp1+exp13 -> $seq_log"
    (
        for key in llama-1b qwen-1.5b gemma-2b; do
            hf_id="${MODELS[$key]}"
            short="${SHORT_NAMES[$key]}"
            outdir_1="results/${short}/exp1"
            outdir_13="results/${short}/exp13"

            echo "=== exp1 ${key} ==="
            $CONDA_RUN python scripts/01_baseline.py \
                --model "$hf_id" \
                --device "cuda:${GPU_SEQ}" \
                --n-test 50 \
                --output-dir "$outdir_1"

            echo "=== exp13 ${key} ==="
            $CONDA_RUN python scripts/13_instance_analysis.py \
                --model "$hf_id" \
                --device "cuda:${GPU_SEQ}" \
                --n-test 20 \
                --output-dir "$outdir_13"
        done
    ) > "$seq_log" 2>&1 &
    PIDS+=($!)
fi

# ---------------------------------------------------------------------------
# Wait for all background jobs
# ---------------------------------------------------------------------------
if ! $DRY_RUN; then
    echo ""
    echo "All jobs launched (${#PIDS[@]} background processes)."
    echo "PIDs: ${PIDS[*]}"
    echo "Logs: $LOG_DIR/"
    echo ""
    echo "Waiting for completion..."
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "FAILED: PID $pid"
            FAIL=$((FAIL + 1))
        fi
    done
    echo ""
    if [[ $FAIL -gt 0 ]]; then
        echo "WARNING: $FAIL job(s) failed. Check logs in $LOG_DIR/"
        exit 1
    else
        echo "All jobs completed successfully."
    fi
fi
