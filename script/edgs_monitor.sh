#!/bin/bash
# EDGS PSNR Monitor - runs until PSNR > MIN_PSNR
set -e

INPUT_VIDEO="$1"
OUTPUT_PATH="$2"
FLAG_360="$3"
MIN_PSNR="$4"
MONITOR_INTERVAL="$5"
MAX_RETRIES="$6"
TRAIN_LOG="$7"
USER_CONFIG="$8"
USER_COLMAP_CONFIG="$9"

PROJECT_DIR="/home/mas/proj/sensyn/EDGS"
cd "$PROJECT_DIR"

# Config arrays - progression from user-specified to fallbacks
DEFAULT_CONFIG="${USER_CONFIG:-train_large}"
DEFAULT_COLMAP_CONFIG="${USER_COLMAP_CONFIG:-colmap_06_lowest_quality}"
CONFIGS=("$DEFAULT_CONFIG" "train_xlarge" "train_06_lowest_quality" "train_06_lowest_quality")
COLMAP_CONFIGS=("$DEFAULT_COLMAP_CONFIG" "$DEFAULT_COLMAP_CONFIG" "colmap_06_lowest_quality" "colmap_06_lowest_quality")

get_psnr() {
    local log="$1"
    # Get the last test PSNR evaluation (evaluations happen at non-round iterations like 29228)
    grep -E "Evaluating test:.*PSNR=" "$log" 2>/dev/null | tail -1 | grep -oP "PSNR=\K[0-9.]+" || echo "0"
}

get_ply_size() {
    local output="$1"
    ls -l "$output/point_cloud/iteration_30000/point_cloud.ply" 2>/dev/null | awk '{print $5}' || echo "0"
}

is_training_running() {
    ps aux | grep -v grep | grep "fit_model_to_scene" | grep -q "$OUTPUT_PATH" 2>/dev/null
}

wait_for_completion() {
    echo "$(date): Waiting for training to complete..."
    while is_training_running; do
        sleep 60
    done
    echo "$(date): Training process ended"
}

run_training() {
    local config="$1"
    local colmap_config="$2"
    local attempt="$3"

    echo "$(date): Starting training attempt $attempt with config=$config, colmap=$colmap_config"

    # Clean previous training artifacts but keep COLMAP if possible
    rm -rf "$OUTPUT_PATH/point_cloud" "$OUTPUT_PATH/chkpnt"* "$OUTPUT_PATH/input.ply" "$OUTPUT_PATH/cameras.json" 2>/dev/null || true

    # If switching to a different colmap config, clean everything
    if [ "$colmap_config" != "${COLMAP_CONFIGS[0]}" ]; then
        rm -rf "$OUTPUT_PATH" 2>/dev/null || true
    fi

    # Run training
    docker compose exec -T edgs-app python script/fit_model_to_scene_full.py \
        --input "/EDGS/${INPUT_VIDEO}" \
        --output_path "/EDGS/${OUTPUT_PATH}" \
        --config "$config" \
        --colmap_config "$colmap_config" \
        $FLAG_360 >> "$TRAIN_LOG" 2>&1 &

    sleep 5  # Wait for process to start
}

check_for_oom() {
    local log="$1"
    grep -q "OutOfMemoryError\|CUDA out of memory" "$log" 2>/dev/null
}

# Main loop
attempt=0
config_idx=0

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    config="${CONFIGS[$config_idx]}"
    colmap_config="${COLMAP_CONFIGS[$config_idx]}"

    echo ""
    echo "=============================================="
    echo "$(date): ATTEMPT $attempt / $MAX_RETRIES"
    echo "Config: $config, COLMAP: $colmap_config"
    echo "=============================================="

    # Start training
    run_training "$config" "$colmap_config" "$attempt"

    # Monitor loop
    while true; do
        sleep $MONITOR_INTERVAL

        if ! is_training_running; then
            echo "$(date): Training ended, checking results..."
            break
        fi

        # Quick progress check
        if [ -f "$TRAIN_LOG" ]; then
            latest=$(grep -E "ITER [0-9]+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oP "ITER \K[0-9]+" || echo "0")
            echo "$(date): Training in progress... iteration $latest"
        fi
    done

    # Check for OOM error
    if check_for_oom "$TRAIN_LOG"; then
        echo "$(date): OOM detected! Switching to lower memory config..."
        config_idx=$((config_idx + 1))
        if [ $config_idx -ge ${#CONFIGS[@]} ]; then
            echo "$(date): ERROR: All configs exhausted, cannot reduce memory further!"
            config_idx=$((${#CONFIGS[@]} - 1))  # Stay at lowest
        fi
        continue
    fi

    # Check PSNR
    psnr=$(get_psnr "$TRAIN_LOG")
    ply_size=$(get_ply_size "$OUTPUT_PATH")

    echo "$(date): Results - PSNR: $psnr, PLY size: $ply_size bytes"

    # Check if PSNR meets threshold
    if [ -n "$psnr" ] && [ "$psnr" != "0" ]; then
        passed=$(echo "$psnr >= $MIN_PSNR" | bc -l 2>/dev/null || echo "0")
        if [ "$passed" = "1" ]; then
            echo ""
            echo "=============================================="
            echo "$(date): SUCCESS! PSNR=$psnr >= $MIN_PSNR"
            echo "Output: $OUTPUT_PATH"
            echo "PLY size: $ply_size bytes"
            echo "=============================================="
            exit 0
        fi
    fi

    # PSNR too low or missing, analyze and retry
    echo "$(date): PSNR=$psnr < $MIN_PSNR, analyzing failure..."

    # If PLY is very small, likely init failed
    if [ "$ply_size" -lt 1000000 ]; then
        echo "$(date): PLY file too small ($ply_size bytes), possible init failure"
        # Don't change config, just retry
    else
        echo "$(date): Training completed but quality insufficient"
        # Keep same config but retry
    fi
done

echo ""
echo "=============================================="
echo "$(date): FAILED after $MAX_RETRIES attempts"
echo "Best PSNR achieved: $psnr"
echo "=============================================="
exit 1
