#!/bin/bash
# EDGS Quality Monitor - retries training until quality thresholds are met
#
# Each retry uses a DIFFERENT strategy based on failure analysis:
#   - Checks train vs test gap (overfitting detection)
#   - Checks gaussian count (capacity issues)
#   - Adjusts iterations, densification, correspondences, loss weights, LR, etc.
#   - Never repeats the same parameter combination
#
# Usage (inside Docker container):
#   bash script/edgs_monitor.sh <input_video> <output_path> [options]
#
# Options:
#   --360                   Enable 360 mode
#   --colmap-config NAME    COLMAP config name (default: colmap_08_360_2fps)
#   --train-config NAME     Base training config (default: train_04_medium_quality)
#   --min-psnr VALUE        Minimum test PSNR (default: 25.0)
#   --min-ssim VALUE        Minimum test SSIM (default: 0.80)
#   --max-lpips VALUE       Maximum test LPIPS (default: 0.18)
#   --max-retries N         Maximum retry attempts (default: 10)
#
# Example:
#   bash script/edgs_monitor.sh /EDGS/data/video.mp4 /EDGS/outputs/scene --360

set -euo pipefail

# --- Parse arguments ---
INPUT_VIDEO="${1:?Usage: $0 <input_video> <output_path> [options]}"
OUTPUT_PATH="${2:?Usage: $0 <input_video> <output_path> [options]}"
shift 2

FLAG_360=""
COLMAP_CONFIG="colmap_08_360_2fps"
TRAIN_CONFIG="train_04_medium_quality"
MIN_PSNR=25.0
MIN_SSIM=0.80
MAX_LPIPS=0.18
MAX_RETRIES=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --360) FLAG_360="--360"; shift ;;
        --colmap-config) COLMAP_CONFIG="$2"; shift 2 ;;
        --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
        --min-psnr) MIN_PSNR="$2"; shift 2 ;;
        --min-ssim) MIN_SSIM="$2"; shift 2 ;;
        --max-lpips) MAX_LPIPS="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Retry configuration ---
# 10 distinct strategies, each with unique parameter combinations.
# Every level differs from ALL previous levels in at least one parameter.
#
# Dimensions varied across levels:
#   - Iterations: 30k / 60k / 90k / 120k
#   - Densification: on (default 500-15k) / extended (500-30k) / off
#   - Correspondences: default (8k,120) / medium (15k,180) / high (20k,240,5nns)
#   - Loss: lambda_dssim 0.2 / 0.4 / 0.5
#   - Densify params: percent_dense, grad_threshold
#   - Opacity reset: 30k (default) / 3000 (frequent)
#   - SfM init: off / on (combine RoMa + SfM points)

ITER_30K=""
ITER_60K="train.gs_epochs=60000 gs.opt.iterations=60000 gs.opt.position_lr_max_steps=60000"
ITER_90K="train.gs_epochs=90000 gs.opt.iterations=90000 gs.opt.position_lr_max_steps=90000"
ITER_120K="train.gs_epochs=120000 gs.opt.iterations=120000 gs.opt.position_lr_max_steps=120000"

SAVE_60K="gs.opt.save_iterations=[3000,7000,15000,30000,60000]"
SAVE_90K="gs.opt.save_iterations=[3000,7000,15000,30000,60000,90000]"
SAVE_120K="gs.opt.save_iterations=[3000,7000,15000,30000,60000,90000,120000]"

NO_DENSIFY="train.no_densify=True"
DENSIFY_EXTENDED="gs.opt.densify_until_iter=30000"
DENSIFY_LONG="gs.opt.densify_until_iter=45000"
DENSIFY_GENTLE="gs.opt.percent_dense=0.005 gs.opt.densify_grad_threshold=0.0001"

CORR_MEDIUM="init_wC.matches_per_ref=15000 init_wC.num_refs=180"
CORR_HIGH="init_wC.matches_per_ref=20000 init_wC.num_refs=240 init_wC.nns_per_ref=5"

DSSIM_04="gs.opt.lambda_dssim=0.4"
DSSIM_05="gs.opt.lambda_dssim=0.5"

OPACITY_FREQ="gs.opt.opacity_reset_interval=3000"

SFM_INIT="init_wC.add_SfM_init=True"

retry_overrides() {
    local level=$1
    case $level in
        0)
            # Baseline: 30k iters, standard densification (500-15k)
            echo ""
            ;;
        1)
            # More iterations, standard densification
            echo "$ITER_60K $SAVE_60K"
            ;;
        2)
            # No densification — diagnostic fallback (only no-densify level)
            echo "$ITER_60K $SAVE_60K $NO_DENSIFY"
            ;;
        3)
            # Extended densification window (to 30k) + more correspondences
            echo "$ITER_60K $SAVE_60K $DENSIFY_EXTENDED $CORR_MEDIUM"
            ;;
        4)
            # 90k iters, standard densification, more correspondences, perceptual loss
            echo "$ITER_90K $SAVE_90K $CORR_MEDIUM $DSSIM_04"
            ;;
        5)
            # 90k iters, extended densification + more correspondences + perceptual
            echo "$ITER_90K $SAVE_90K $DENSIFY_EXTENDED $CORR_MEDIUM $DSSIM_04"
            ;;
        6)
            # 90k iters, extended densification + max correspondences + SfM init
            echo "$ITER_90K $SAVE_90K $DENSIFY_EXTENDED $CORR_HIGH $SFM_INIT"
            ;;
        7)
            # 120k iters, standard densification, max correspondences, high DSSIM
            echo "$ITER_120K $SAVE_120K $CORR_HIGH $DSSIM_05"
            ;;
        8)
            # 120k iters, extended densification + max corr + SfM + high DSSIM
            echo "$ITER_120K $SAVE_120K $DENSIFY_EXTENDED $CORR_HIGH $DSSIM_05 $SFM_INIT"
            ;;
        9)
            # Maximum effort: long gentle densification + max corr + SfM + high DSSIM + freq reset
            echo "$ITER_120K $SAVE_120K $DENSIFY_LONG $DENSIFY_GENTLE $CORR_HIGH $DSSIM_05 $SFM_INIT $OPACITY_FREQ"
            ;;
    esac
}

retry_description() {
    local level=$1
    case $level in
        0) echo "baseline (30k, densify)" ;;
        1) echo "60k, densify" ;;
        2) echo "60k, no densify (diagnostic)" ;;
        3) echo "60k, extended densify(30k), more corr(15k/180)" ;;
        4) echo "90k, densify, more corr(15k/180), DSSIM(0.4)" ;;
        5) echo "90k, extended densify(30k), more corr, DSSIM(0.4)" ;;
        6) echo "90k, extended densify(30k), max corr(20k/240/5nns), SfM init" ;;
        7) echo "120k, densify, max corr, DSSIM(0.5)" ;;
        8) echo "120k, extended densify(30k), max corr, SfM, DSSIM(0.5)" ;;
        9) echo "120k, long densify(45k), gentle, max corr, SfM, DSSIM(0.5), freq reset" ;;
    esac
}

# --- Diagnostic function ---
# Analyzes training log to understand failure mode
diagnose_failure() {
    local log="$1"

    echo ""
    echo "--- Diagnosis ---"

    # Extract train metrics
    local last_train
    last_train=$(grep -E "Evaluating train:.*PSNR=" "$log" 2>/dev/null | tail -1) || true
    local last_test
    last_test=$(grep -E "Evaluating test:.*PSNR=" "$log" 2>/dev/null | tail -1) || true

    if [ -z "$last_test" ]; then
        echo "  No test metrics found — training may have crashed"
        return
    fi

    # Parse test metrics
    local test_psnr test_ssim test_lpips
    test_psnr=$(echo "$last_test" | grep -oP 'PSNR=\K[0-9.]+') || test_psnr="0"
    test_ssim=$(echo "$last_test" | grep -oP 'SSIM=\K[0-9.]+') || test_ssim="0"
    test_lpips=$(echo "$last_test" | grep -oP 'LPIPS_splat=\K[0-9.]+') || test_lpips="1"

    # Parse train metrics if available
    if [ -n "$last_train" ]; then
        local train_psnr train_ssim
        train_psnr=$(echo "$last_train" | grep -oP 'PSNR=\K[0-9.]+') || train_psnr="0"
        train_ssim=$(echo "$last_train" | grep -oP 'SSIM=\K[0-9.]+') || train_ssim="0"

        local psnr_gap
        psnr_gap=$(python3 -c "print(f'{float(\"$train_psnr\") - float(\"$test_psnr\"):.2f}')")
        echo "  Train PSNR=${train_psnr} vs Test PSNR=${test_psnr} (gap=${psnr_gap}dB)"

        local is_overfitting
        is_overfitting=$(python3 -c "print('YES' if float('$psnr_gap') > 3.0 else 'no')")
        echo "  Overfitting: ${is_overfitting}"
    fi

    # Extract gaussian count
    local gauss_count
    gauss_count=$(echo "$last_test" | grep -oP '#\K[0-9]+(?= gaussians)') || gauss_count="unknown"
    echo "  Gaussians: ${gauss_count}"

    # Identify which metrics are failing and by how much
    local psnr_deficit lpips_deficit
    psnr_deficit=$(python3 -c "d=float('$MIN_PSNR')-float('$test_psnr'); print(f'{d:.2f}' if d>0 else 'PASS')")
    lpips_deficit=$(python3 -c "d=float('$test_lpips')-float('$MAX_LPIPS'); print(f'{d:.3f}' if d>0 else 'PASS')")
    local ssim_deficit
    ssim_deficit=$(python3 -c "d=float('$MIN_SSIM')-float('$test_ssim'); print(f'{d:.3f}' if d>0 else 'PASS')")

    echo "  PSNR deficit: ${psnr_deficit}"
    echo "  SSIM deficit: ${ssim_deficit}"
    echo "  LPIPS excess: ${lpips_deficit}"
}

# --- Quality check function ---
check_quality() {
    local log="$1"

    local last_test
    last_test=$(grep -E "Evaluating test:.*PSNR=" "$log" 2>/dev/null | tail -1) || true

    if [ -z "$last_test" ]; then
        echo "NO_METRICS"
        return 1
    fi

    local psnr ssim lpips
    psnr=$(echo "$last_test" | grep -oP 'PSNR=\K[0-9.]+') || psnr="0"
    ssim=$(echo "$last_test" | grep -oP 'SSIM=\K[0-9.]+') || ssim="0"
    lpips=$(echo "$last_test" | grep -oP 'LPIPS_splat=\K[0-9.]+') || lpips="1"

    echo "PSNR=${psnr} SSIM=${ssim} LPIPS=${lpips}"

    local pass_psnr pass_ssim pass_lpips
    pass_psnr=$(python3 -c "print(1 if float('$psnr') >= float('$MIN_PSNR') else 0)")
    pass_ssim=$(python3 -c "print(1 if float('$ssim') >= float('$MIN_SSIM') else 0)")
    pass_lpips=$(python3 -c "print(1 if float('$lpips') <= float('$MAX_LPIPS') else 0)")

    if [ "$pass_psnr" = "1" ] && [ "$pass_ssim" = "1" ] && [ "$pass_lpips" = "1" ]; then
        return 0
    else
        [ "$pass_psnr" = "0" ] && echo "  FAIL: PSNR ${psnr} < ${MIN_PSNR}"
        [ "$pass_ssim" = "0" ] && echo "  FAIL: SSIM ${ssim} < ${MIN_SSIM}"
        [ "$pass_lpips" = "0" ] && echo "  FAIL: LPIPS ${lpips} > ${MAX_LPIPS}"
        return 1
    fi
}

check_for_oom() {
    local log="$1"
    local exit_code="${2:-0}"
    # Standard OOM patterns
    grep -q "OutOfMemoryError\|CUDA out of memory\|Killed" "$log" 2>/dev/null && return 0
    # Watchdog kill markers
    grep -q "VRAM_WATCHDOG_KILL\|RAM_WATCHDOG_KILL" "$log" 2>/dev/null && return 0
    # Exit code 137 = OOM killer
    [ "$exit_code" -eq 137 ] 2>/dev/null && return 0
    # Non-zero exit with no test metrics (likely crashed from resource exhaustion)
    if [ "$exit_code" -ne 0 ] 2>/dev/null; then
        if ! grep -q "Evaluating test:.*PSNR=" "$log" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# --- GPU VRAM watchdog ---
# Monitors VRAM usage and kills the training process if it stays >= 95% for 60s
WATCHDOG_PID=""

start_vram_watchdog() {
    local train_pid="$1"
    local log_file="$2"

    (
        consecutive_high=0
        while kill -0 "$train_pid" 2>/dev/null; do
            sleep 30
            # Get VRAM usage percentage from nvidia-smi
            vram_pct=$(nvidia-smi --query-gpu=memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | \
                awk -F',' '{printf "%.0f", ($1/$2)*100}') || continue
            if [ "$vram_pct" -ge 95 ] 2>/dev/null; then
                consecutive_high=$((consecutive_high + 1))
                if [ "$consecutive_high" -ge 2 ]; then
                    echo "[VRAM_WATCHDOG_KILL] VRAM at ${vram_pct}% for 60s+ — killing PID ${train_pid}" >> "$log_file"
                    kill -TERM "$train_pid" 2>/dev/null || true
                    sleep 5
                    kill -KILL "$train_pid" 2>/dev/null || true
                    exit 0
                fi
            else
                consecutive_high=0
            fi
        done
    ) &
    WATCHDOG_PID=$!
}

stop_vram_watchdog() {
    if [ -n "$WATCHDOG_PID" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
    fi
}

# --- System RAM watchdog ---
# Monitors system RAM usage and kills training if it stays >= 85% for 90s.
# This prevents the system from freezing due to swap thrashing on low-swap systems.
RAM_WATCHDOG_PID=""

start_ram_watchdog() {
    local train_pid="$1"
    local log_file="$2"

    (
        consecutive_high=0
        while kill -0 "$train_pid" 2>/dev/null; do
            sleep 30
            # Get RAM usage percentage from /proc/meminfo
            ram_pct=$(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%.0f", ((t-a)/t)*100}' /proc/meminfo 2>/dev/null) || continue
            if [ "$ram_pct" -ge 85 ] 2>/dev/null; then
                consecutive_high=$((consecutive_high + 1))
                if [ "$consecutive_high" -ge 3 ]; then
                    echo "[RAM_WATCHDOG_KILL] System RAM at ${ram_pct}% for 90s+ — killing PID ${train_pid}" >> "$log_file"
                    kill -TERM "$train_pid" 2>/dev/null || true
                    sleep 5
                    kill -KILL "$train_pid" 2>/dev/null || true
                    exit 0
                fi
            else
                consecutive_high=0
            fi
        done
    ) &
    RAM_WATCHDOG_PID=$!
}

stop_ram_watchdog() {
    if [ -n "$RAM_WATCHDOG_PID" ]; then
        kill "$RAM_WATCHDOG_PID" 2>/dev/null || true
        wait "$RAM_WATCHDOG_PID" 2>/dev/null || true
        RAM_WATCHDOG_PID=""
    fi
}

# --- Best result tracking ---
# Extracts test PSNR from log as a comparable float (returns "0" if unavailable)
get_test_psnr() {
    local log="$1"
    local last_test
    last_test=$(grep -E "Evaluating test:.*PSNR=" "$log" 2>/dev/null | tail -1) || true
    if [ -n "$last_test" ]; then
        echo "$last_test" | grep -oP 'PSNR=\K[0-9.]+' || echo "0"
    else
        echo "0"
    fi
}

# Copies current training result to best directory if PSNR improved
update_best_result() {
    local current_psnr="$1"
    local is_better
    is_better=$(python3 -c "print(1 if float('$current_psnr') > float('$BEST_PSNR') else 0)")

    if [ "$is_better" = "1" ]; then
        echo "  NEW BEST: PSNR ${current_psnr} > previous ${BEST_PSNR}"
        BEST_PSNR="$current_psnr"

        # Copy training artifacts to best directory
        rm -rf "$BEST_DIR" 2>/dev/null || true
        mkdir -p "$BEST_DIR"
        # Copy point clouds and model files
        [ -d "$OUTPUT_PATH/point_cloud" ] && cp -r "$OUTPUT_PATH/point_cloud" "$BEST_DIR/"
        for f in input.ply cameras.json exposure.json train_config.yaml; do
            [ -f "$OUTPUT_PATH/$f" ] && cp "$OUTPUT_PATH/$f" "$BEST_DIR/"
        done
        # Copy training log
        [ -f "$LOG_FILE" ] && cp "$LOG_FILE" "$BEST_LOG"

        echo "  Saved best result to: $BEST_DIR"
    else
        echo "  No improvement: PSNR ${current_psnr} <= best ${BEST_PSNR}"
    fi
}

# --- Main ---
SCENE_NAME=$(basename "$OUTPUT_PATH")
LOG_FILE="${OUTPUT_PATH}_train.log"
BEST_DIR="${OUTPUT_PATH}_best"
BEST_LOG="${OUTPUT_PATH}_best_train.log"
BEST_PSNR="0"

echo "=============================================="
echo "EDGS Quality Monitor"
echo "=============================================="
echo "Input:       $(basename "$INPUT_VIDEO")"
echo "Output:      $OUTPUT_PATH (last) / ${BEST_DIR} (best)"
echo "360 mode:    ${FLAG_360:-no}"
echo "COLMAP:      $COLMAP_CONFIG"
echo "Train base:  $TRAIN_CONFIG"
echo "Thresholds:  PSNR>=${MIN_PSNR}  SSIM>=${MIN_SSIM}  LPIPS<=${MAX_LPIPS}"
echo "Max retries: $MAX_RETRIES"
echo "Log:         $LOG_FILE"
echo "=============================================="

attempt=0
retry_level=0
had_oom=false

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))

    # Note: densification levels are safe after OOM thanks to Gaussian count cap
    # and VRAM guard in trainer.py — no need to skip them

    overrides=$(retry_overrides $retry_level)
    desc=$(retry_description $retry_level)

    echo ""
    echo "=============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S'): ATTEMPT ${attempt}/${MAX_RETRIES} — ${desc}"
    echo "  Overrides: ${overrides:-none}"
    echo "=============================================="

    # Clean previous training artifacts (keep COLMAP scene)
    rm -rf "$OUTPUT_PATH/point_cloud" "$OUTPUT_PATH/chkpnt"* \
           "$OUTPUT_PATH/input.ply" "$OUTPUT_PATH/cameras.json" \
           "$OUTPUT_PATH/exposure.json" "$OUTPUT_PATH/viewpoint_"*.png \
           "$OUTPUT_PATH/train_config.yaml" 2>/dev/null || true

    # Build command
    CMD="python -u script/fit_model_to_scene_full.py"
    CMD="$CMD --input '$INPUT_VIDEO'"
    CMD="$CMD --output_path '$OUTPUT_PATH'"
    CMD="$CMD --config '$TRAIN_CONFIG'"
    CMD="$CMD --colmap_config '$COLMAP_CONFIG'"
    [ -n "$FLAG_360" ] && CMD="$CMD $FLAG_360"
    [ -n "$overrides" ] && CMD="$CMD --overrides $overrides"

    echo "  CMD: $CMD"

    # Run training (capture output to log) with VRAM watchdog
    START_TIME=$(date +%s)
    eval "$CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    start_vram_watchdog "$TRAIN_PID" "$LOG_FILE"
    start_ram_watchdog "$TRAIN_PID" "$LOG_FILE"
    wait "$TRAIN_PID" && EXIT_CODE=0 || EXIT_CODE=$?
    stop_vram_watchdog
    stop_ram_watchdog
    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

    echo "$(date '+%Y-%m-%d %H:%M:%S'): Training exited (code=${EXIT_CODE}) after ${ELAPSED} min"

    # Check for OOM (pass exit code for 137 detection)
    if check_for_oom "$LOG_FILE" "$EXIT_CODE"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): OOM/resource exhaustion detected (exit code=${EXIT_CODE})!"
        had_oom=true
        retry_level=$((retry_level + 1))
        echo "  Advancing to retry level ${retry_level}"
        continue
    fi

    # Check quality
    echo ""
    echo "--- Quality Check ---"
    quality_output=$(check_quality "$LOG_FILE") && quality_status=0 || quality_status=$?
    echo "$quality_output"

    # Track best result across all attempts
    current_psnr=$(get_test_psnr "$LOG_FILE")
    update_best_result "$current_psnr"

    if [ $quality_status -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S'): SUCCESS after ${attempt} attempts!"
        echo "  ${quality_output}"
        echo "  Output: ${OUTPUT_PATH} (last) / ${BEST_DIR} (best)"
        echo "=============================================="
        exit 0
    fi

    # Run diagnostics on failure
    diagnose_failure "$LOG_FILE"

    # Advance to next level (always different strategy)
    retry_level=$((retry_level + 1))
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Quality insufficient, advancing to retry level ${retry_level}"
done

echo ""
echo "=============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S'): EXHAUSTED ${MAX_RETRIES} attempts"
echo "  Last metrics: $(check_quality "$LOG_FILE" 2>/dev/null || echo 'unavailable')"
echo "  Best PSNR: ${BEST_PSNR}"
echo "  Best result: ${BEST_DIR}"
echo "=============================================="
exit 1
