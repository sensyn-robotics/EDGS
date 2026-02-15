#!/bin/bash
#
# Batch Video Processing Script for EDGS
# Processes all videos in a directory with automatic OOM retry using lower quality configs
#
# Usage:
#   ./script/batch_process_videos.sh [input_dir] [output_base]
#
# Examples:
#   ./script/batch_process_videos.sh data/20260121_otowa outputs/20260121_otowa
#   nohup ./script/batch_process_videos.sh > outputs/20260121_otowa/batch_process.log 2>&1 &
#

set -euo pipefail

# Default paths (can be overridden by arguments)
INPUT_DIR="${1:-data/20260121_otowa}"
OUTPUT_BASE="${2:-outputs/20260121_otowa}"

# Config levels for retry (train_config:colmap_config)
# Strategy: Start with high quality COLMAP (more frames = better registration)
# then fall back to lower quality on OOM
CONFIGS=(
    "train:colmap_02_high_quality"
    "train:colmap_01_highest_quality"
    "train_04_medium_quality:colmap_03_optimal_quality"
    "train_05_low_quality:colmap_04_medium_quality"
    "train_06_lowest_quality:colmap_05_low_quality"
)

# Container name
CONTAINER_NAME="edgs-app"

# Timestamp for logging
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

log_error() {
    echo "[$(timestamp)] ERROR: $*" >&2
}

# Check if docker compose is available
check_docker() {
    if ! docker compose ps --services 2>/dev/null | grep -q "edgs-app"; then
        log_error "Docker container 'edgs-app' is not running. Start it with: docker compose up -d"
        exit 1
    fi
    log "Docker container '$CONTAINER_NAME' is running"
}

# Check if processing is complete for a video
is_complete() {
    local output_dir="$1"
    # Check for point_cloud directory which indicates successful completion
    if [ -d "$output_dir/point_cloud" ]; then
        return 0
    fi
    return 1
}

# Check if output contains OOM error
check_oom_error() {
    local output="$1"
    if echo "$output" | grep -qiE "out of memory|CUDA out of memory|OOM|RuntimeError.*memory|torch.cuda.OutOfMemoryError"; then
        return 0
    fi
    return 1
}

# Check if output contains COLMAP reconstruction failure
check_colmap_failure() {
    local output="$1"
    if echo "$output" | grep -qiE "Discarding reconstruction|insufficient size|Could not find good initial|min_num_inliers|Failed to reconstruct"; then
        return 0
    fi
    return 1
}

# Process a single video
process_video() {
    local video_path="$1"
    local video_file
    video_file=$(basename "$video_path")
    local basename="${video_file%.*}"  # Remove extension
    local output_dir="$OUTPUT_BASE/$basename"

    log "=========================================="
    log "Processing: $video_file"
    log "Output: $output_dir"
    log "=========================================="

    # Skip if already completed
    if is_complete "$output_dir"; then
        log "SKIPPED: $basename (already completed - point_cloud exists)"
        return 0
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Try each config level on OOM
    local config_level=0
    for config_pair in "${CONFIGS[@]}"; do
        config_level=$((config_level + 1))

        # Parse train and colmap configs
        local train_config="${config_pair%%:*}"
        local colmap_config="${config_pair##*:}"

        log "Attempt $config_level/${#CONFIGS[@]}: train=$train_config, colmap=$colmap_config"

        # Clean up previous failed attempt (but preserve logs)
        if [ -d "$output_dir/sparse" ] || [ -d "$output_dir/images" ] || [ -d "$output_dir/video_scene" ]; then
            log "Cleaning up previous failed attempt..."
            # Keep log files but remove processing artifacts
            find "$output_dir" -mindepth 1 -maxdepth 1 \
                ! -name "*.log" ! -name "*.txt" \
                -exec rm -rf {} + 2>/dev/null || true
        fi

        # Run processing inside Docker container
        # Quote paths to handle special characters like parentheses
        # Add --360 flag for equirectangular 360 video processing
        local cmd="python script/fit_model_to_scene_full.py \
            --input '/EDGS/$video_path' \
            --output_path '/EDGS/$output_dir' \
            --config $train_config \
            --colmap_config $colmap_config \
            --360"

        log "Running: docker compose exec $CONTAINER_NAME $cmd"

        # Execute and capture output
        local output_file="$output_dir/attempt_${config_level}.log"
        local exit_code=0

        # Run with timeout (8 hours max per video) and capture output
        if timeout 28800 docker compose exec -T "$CONTAINER_NAME" bash -c "$cmd" > "$output_file" 2>&1; then
            log "SUCCESS: $basename completed with config level $config_level"
            return 0
        else
            exit_code=$?
        fi

        # Check error type and decide retry strategy
        local log_content
        log_content=$(cat "$output_file")

        if check_oom_error "$log_content"; then
            log "OOM detected at config level $config_level. Will retry with lower quality..."
        elif check_colmap_failure "$log_content"; then
            log "COLMAP reconstruction failed at config level $config_level. Will retry with different settings..."
        else
            # Unknown error - still retry with next config
            log "Unknown error at config level $config_level (exit code: $exit_code). Will retry..."
        fi

        # If this was the last config level, fail
        if [ $config_level -eq ${#CONFIGS[@]} ]; then
            log_error "FAILED: $basename - Exhausted all config levels"
            log_error "Check log file: $output_file"
            return 1
        fi
        # Continue to next config level
        continue
    done

    # Should not reach here, but handle gracefully
    log_error "FAILED: $basename - Unexpected exit from retry loop"
    return 1
}

# Main execution
main() {
    log "=========================================="
    log "EDGS Batch Video Processing"
    log "=========================================="
    log "Input directory: $INPUT_DIR"
    log "Output base: $OUTPUT_BASE"
    log "Config levels: ${#CONFIGS[@]}"
    log ""

    # Check prerequisites
    check_docker

    # Ensure output base directory exists
    mkdir -p "$OUTPUT_BASE"

    # Find all video files and sort them
    local videos=()
    while IFS= read -r -d '' video; do
        videos+=("$video")
    done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" \) -print0 | sort -z)

    local total=${#videos[@]}
    log "Found $total video(s) to process"
    log ""

    if [ $total -eq 0 ]; then
        log_error "No video files found in $INPUT_DIR"
        exit 1
    fi

    # Process each video
    local processed=0
    local skipped=0
    local failed=0
    local failed_videos=()

    for video in "${videos[@]}"; do
        if process_video "$video"; then
            if is_complete "$OUTPUT_BASE/$(basename "${video%.*}")"; then
                processed=$((processed + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            failed=$((failed + 1))
            failed_videos+=("$(basename "$video")")
        fi
        log ""
    done

    # Summary
    log "=========================================="
    log "BATCH PROCESSING COMPLETE"
    log "=========================================="
    log "Total videos: $total"
    log "Successfully processed: $processed"
    log "Previously completed (skipped): $skipped"
    log "Failed: $failed"

    if [ $failed -gt 0 ]; then
        log ""
        log "Failed videos:"
        for v in "${failed_videos[@]}"; do
            log "  - $v"
        done
    fi

    # Exit with error if any failed
    if [ $failed -gt 0 ]; then
        exit 1
    fi
}

# Run main
main "$@"
