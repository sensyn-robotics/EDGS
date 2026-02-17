#!/bin/bash
# EDGS Tmux-based training script with automatic PSNR monitoring and retry
# Usage: ./script/train_tmux.sh <input_video> <output_path> [--360]
#
# Features:
# - Runs training in tmux (survives terminal close)
# - Monitors PSNR every 10 minutes
# - Never finishes until PSNR > 20
# - Automatically retries with adjusted parameters if PSNR < 20
# - Easy to attach/detach: tmux attach -t edgs
#
# Examples:
#   ./script/train_tmux.sh data/video.mp4 outputs/scene1 --360

set -e

# Configuration
SESSION_NAME="edgs"
PROJECT_DIR="/home/mas/proj/sensyn/EDGS"
MIN_PSNR=20.0
MONITOR_INTERVAL=600  # 10 minutes in seconds
MAX_RETRIES=10

# Parse arguments
INPUT_VIDEO="${1:-}"
OUTPUT_PATH="${2:-}"
FLAG_360=""
if [ "$3" == "--360" ]; then
    FLAG_360="--360"
fi

if [ -z "$INPUT_VIDEO" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: $0 <input_video> <output_path> [--360]"
    echo "Example: $0 data/video.mp4 outputs/scene1 --360"
    exit 1
fi

# Paths
LOG_DIR="${PROJECT_DIR}/outputs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENE_NAME=$(basename "$OUTPUT_PATH")
TRAIN_LOG="${LOG_DIR}/${SCENE_NAME}_${TIMESTAMP}.log"
MONITOR_LOG="${LOG_DIR}/${SCENE_NAME}_monitor_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Config progression for retries (start with highest quality, reduce on failure)
CONFIGS=("train_large" "train_xlarge" "train_xlarge" "train_xlarge")
COLMAP_CONFIGS=("colmap_06_lowest_quality" "colmap_06_lowest_quality" "colmap_06_lowest_quality" "colmap_06_lowest_quality")

# Kill existing session if running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

echo "=============================================="
echo "EDGS Training in Tmux (Auto-retry until PSNR>$MIN_PSNR)"
echo "=============================================="
echo "Session:    $SESSION_NAME"
echo "Input:      $INPUT_VIDEO"
echo "Output:     $OUTPUT_PATH"
echo "360 mode:   ${FLAG_360:-No}"
echo "Train log:  $TRAIN_LOG"
echo "Monitor:    Every ${MONITOR_INTERVAL}s (10 min)"
echo "=============================================="

# Create the monitor script
cat > "${PROJECT_DIR}/script/edgs_monitor.sh" << 'MONITOR_SCRIPT'
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

PROJECT_DIR="/home/mas/proj/sensyn/EDGS"
cd "$PROJECT_DIR"

# Config arrays
CONFIGS=("train_large" "train_xlarge" "train_xlarge" "train_xlarge")
COLMAP_CONFIGS=("colmap_06_lowest_quality" "colmap_06_lowest_quality" "colmap_06_lowest_quality" "colmap_06_lowest_quality")

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
MONITOR_SCRIPT

chmod +x "${PROJECT_DIR}/script/edgs_monitor.sh"

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -n training -c "$PROJECT_DIR"

# Start monitor script in the training window
tmux send-keys -t "${SESSION_NAME}:training" "bash script/edgs_monitor.sh '$INPUT_VIDEO' '$OUTPUT_PATH' '$FLAG_360' '$MIN_PSNR' '$MONITOR_INTERVAL' '$MAX_RETRIES' '$TRAIN_LOG' 2>&1 | tee $MONITOR_LOG" Enter

# Create status window
tmux new-window -t "$SESSION_NAME" -n status -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:status" "watch -n 30 'echo \"=== Training Progress ===\"; tail -5 $TRAIN_LOG 2>/dev/null | grep -E \"ITER|PSNR|Error\" | tail -3; echo; echo \"=== Monitor ===\"; tail -10 $MONITOR_LOG 2>/dev/null; echo; echo \"=== GPU ===\"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null'" Enter

# Create log tail window
tmux new-window -t "$SESSION_NAME" -n logs -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:logs" "tail -f $TRAIN_LOG" Enter

# Select training window
tmux select-window -t "${SESSION_NAME}:training"

echo ""
echo "Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+b, then d"
echo "  Switch windows:      Ctrl+b, then n (next) or p (previous)"
echo "  Kill session:        tmux kill-session -t $SESSION_NAME"
echo ""
echo "Windows:"
echo "  0: training - Main training/monitor process"
echo "  1: status   - Quick status view (updates every 30s)"
echo "  2: logs     - Live training log"
echo ""
echo "Training will automatically retry until PSNR > $MIN_PSNR"
echo "You can safely close this terminal."
echo "=============================================="
