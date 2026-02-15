#!/bin/bash
# Batch retrain all scenes until PSNR > 20
# This script processes scenes sequentially, retrying each until success

set -e
cd /home/mas/proj/sensyn/EDGS

LOG_DIR="outputs/20260121_otowa"
MIN_PSNR=20.0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="${LOG_DIR}/batch_retrain_${TIMESTAMP}.log"

# Config progression
CONFIGS=("train" "train_medium" "train_large" "train_xlarge")
COLMAP_CONFIGS=("colmap_05_low_quality" "colmap_05_low_quality" "colmap_06_lowest_quality" "colmap_06_lowest_quality")

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$BATCH_LOG"
}

get_psnr() {
    local log="$1"
    grep -E "ITER (29|30)000.*test.*PSNR" "$log" 2>/dev/null | tail -1 | grep -oP "PSNR=\K[0-9.]+" || echo "0"
}

get_ply_size() {
    local output="$1"
    ls -l "$output/point_cloud/iteration_30000/point_cloud.ply" 2>/dev/null | awk '{print $5}' || echo "0"
}

check_oom() {
    local log="$1"
    grep -q "OutOfMemoryError\|CUDA out of memory" "$log" 2>/dev/null
}

wait_for_training() {
    while ps aux | grep -v grep | grep "fit_model_to_scene" > /dev/null 2>&1; do
        sleep 60
    done
}

train_scene() {
    local input="$1"
    local output="$2"
    local scene_name="$3"
    local max_attempts=10

    log "=========================================="
    log "Processing: $scene_name"
    log "=========================================="

    local attempt=0
    local config_idx=0

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        local config="${CONFIGS[$config_idx]}"
        local colmap_config="${COLMAP_CONFIGS[$config_idx]}"
        local train_log="${LOG_DIR}/${scene_name}_attempt${attempt}.log"

        log "Attempt $attempt: config=$config, colmap=$colmap_config"

        # Clean previous artifacts
        rm -rf "$output/point_cloud" "$output/chkpnt"* "$output/input.ply" "$output/cameras.json" 2>/dev/null || true

        # If using lower colmap config, clean everything
        if [ $config_idx -ge 2 ]; then
            rm -rf "$output" 2>/dev/null || true
        fi

        # Run training
        docker compose exec -T edgs-app python script/fit_model_to_scene_full.py \
            --input "/EDGS/$input" \
            --output_path "/EDGS/$output" \
            --config "$config" \
            --colmap_config "$colmap_config" \
            --360 > "$train_log" 2>&1 &

        sleep 10
        wait_for_training

        # Check for OOM
        if check_oom "$train_log"; then
            log "OOM detected, reducing memory settings..."
            config_idx=$((config_idx + 1))
            if [ $config_idx -ge ${#CONFIGS[@]} ]; then
                config_idx=$((${#CONFIGS[@]} - 1))
            fi
            continue
        fi

        # Check PSNR
        local psnr=$(get_psnr "$train_log")
        local ply_size=$(get_ply_size "$output")

        log "Result: PSNR=$psnr, PLY size=$ply_size"

        if [ -n "$psnr" ] && [ "$psnr" != "0" ]; then
            local passed=$(echo "$psnr >= $MIN_PSNR" | bc -l 2>/dev/null || echo "0")
            if [ "$passed" = "1" ]; then
                log "SUCCESS: $scene_name PSNR=$psnr >= $MIN_PSNR"
                return 0
            fi
        fi

        log "PSNR too low, retrying..."

        # If PLY very small, might need different config
        if [ "$ply_size" -lt 100000 ]; then
            config_idx=$((config_idx + 1))
            if [ $config_idx -ge ${#CONFIGS[@]} ]; then
                config_idx=$((${#CONFIGS[@]} - 1))
            fi
        fi
    done

    log "FAILED: $scene_name after $max_attempts attempts"
    return 1
}

# Main
log "=========================================="
log "EDGS Batch Retrain - All Scenes"
log "Target PSNR: >= $MIN_PSNR"
log "=========================================="

# Wait for any current training to finish
log "Waiting for current training to complete..."
wait_for_training

# Scenes to process (those with PSNR < 20 or broken)
# Format: input_video output_path scene_name

# 2-1: Currently running, will check after
# 1_1: PSNR 19.25 < 20
# 3-1: PSNR 17.26 < 20
# 4: Broken (29K PLY)

# Check 2-1 result first
PSNR_2_1=$(get_psnr "${LOG_DIR}/2-1_retrain.log")
log "2-1 current PSNR: $PSNR_2_1"
if [ -z "$PSNR_2_1" ] || [ "$PSNR_2_1" = "0" ] || [ $(echo "$PSNR_2_1 < $MIN_PSNR" | bc -l) = "1" ]; then
    train_scene "data/20260121_otowa/2-1.細めの1本（No.47）を離隔1~2mで一周_VID_20260121_115814_00_079.mp4" \
                "outputs/20260121_otowa/2-1.細めの1本（No.47）を離隔1~2mで一周_VID_20260121_115814_00_079" \
                "2-1"
fi

# 1_1
train_scene "data/20260121_otowa/1_1.太めの1本（No.41）を離隔2mで一周_VID_20260121_120105_00_081.mp4" \
            "outputs/20260121_otowa/1_1.太めの1本（No.41）を離隔2mで一周_VID_20260121_120105_00_081" \
            "1_1"

# 3-1
train_scene "data/20260121_otowa/3-1.細めの4本(No.43-46)を離隔1mで一周_VID_20260121_114759_00_077.mp4" \
            "outputs/20260121_otowa/3-1.細めの4本(No.43-46)を離隔1mで一周_VID_20260121_114759_00_077" \
            "3-1"

# 4
train_scene "data/20260121_otowa/4.検証エリア全体（約20本）を運用を想定したジグザグ移動で撮影_VID_20260121_114006_00_074.mp4" \
            "outputs/20260121_otowa/4.検証エリア全体（約20本）を運用を想定したジグザグ移動で撮影_VID_20260121_114006_00_074" \
            "4"

log "=========================================="
log "BATCH COMPLETE"
log "=========================================="

# Final summary
log "Final Results:"
for dir in ${LOG_DIR}/*/; do
    name=$(basename "$dir" | cut -c1-10)
    ply_size=$(get_ply_size "$dir")
    # Find latest log
    latest_log=$(ls -t "${LOG_DIR}/${name}"*.log 2>/dev/null | head -1)
    psnr=$(get_psnr "$latest_log")
    log "$name: PSNR=$psnr, PLY=$ply_size"
done
