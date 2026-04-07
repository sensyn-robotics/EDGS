#!/bin/bash
# EDGS Tmux-based training with automatic quality retry
#
# Single video:
#   ./script/train_tmux.sh --input data/video.mp4 --output_path outputs/scene1 --360
#
# Batch (all videos in a directory):
#   ./script/train_tmux.sh --batch --data_dir data/20260121_otowa --output_base outputs/20260121_otowa --360
#
# Features:
# - Runs in tmux (survives terminal close)
# - Retries with different strategies until PSNR>=25, SSIM>=0.80, LPIPS<=0.18
# - Skips scenes that already meet quality thresholds
# - Easy: tmux attach -t edgs

set -e

SESSION_NAME="edgs"
PROJECT_DIR="/home/mas/proj/sensyn/EDGS"
MAX_RETRIES=10

# Parse arguments
BATCH_MODE=false
INPUT_VIDEO=""
OUTPUT_PATH=""
DATA_DIR=""
OUTPUT_BASE=""
FLAG_360=""
USER_CONFIG=""
USER_COLMAP_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)         BATCH_MODE=true; shift ;;
        --input)         INPUT_VIDEO="$2"; shift 2 ;;
        --output_path)   OUTPUT_PATH="$2"; shift 2 ;;
        --data_dir)      DATA_DIR="$2"; shift 2 ;;
        --output_base)   OUTPUT_BASE="$2"; shift 2 ;;
        --360)           FLAG_360="--360"; shift ;;
        --config)        USER_CONFIG="$2"; shift 2 ;;
        --colmap_config) USER_COLMAP_CONFIG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

DEFAULT_CONFIG="${USER_CONFIG:-train_04_medium_quality}"
DEFAULT_COLMAP_CONFIG="${USER_COLMAP_CONFIG:-colmap_08_360_2fps}"

# Validate
if [ "$BATCH_MODE" = true ]; then
    if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_BASE" ]; then
        echo "Batch usage: $0 --batch --data_dir <dir> --output_base <dir> [--360] [--config <name>] [--colmap_config <name>]"
        exit 1
    fi
else
    if [ -z "$INPUT_VIDEO" ] || [ -z "$OUTPUT_PATH" ]; then
        echo "Single usage: $0 --input <video> --output_path <dir> [--360] [--config <name>] [--colmap_config <name>]"
        exit 1
    fi
fi

# Kill existing session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# Also kill any running training processes in Docker
docker compose exec -T edgs-app bash -c "pkill -f 'batch_train_360' 2>/dev/null; pkill -f 'edgs_monitor' 2>/dev/null; pkill -f 'fit_model_to_scene_full' 2>/dev/null" 2>/dev/null || true
sleep 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$BATCH_MODE" = true ]; then
    # --- Batch mode ---
    MONITOR_LOG="${PROJECT_DIR}/outputs/logs/batch_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$MONITOR_LOG")"

    # Map host paths to Docker paths
    DOCKER_DATA_DIR="/EDGS/data/$(basename "$DATA_DIR")"
    DOCKER_OUTPUT_BASE="/EDGS/outputs/$(basename "$OUTPUT_BASE")"

    DOCKER_CMD="bash /EDGS/script/batch_train_360.sh '${DOCKER_DATA_DIR}' '${DOCKER_OUTPUT_BASE}'"

    echo "=============================================="
    echo "EDGS Batch Training in Tmux"
    echo "=============================================="
    echo "Session:    $SESSION_NAME"
    echo "Data dir:   $DATA_DIR -> $DOCKER_DATA_DIR"
    echo "Output:     $OUTPUT_BASE -> $DOCKER_OUTPUT_BASE"
    echo "360 mode:   ${FLAG_360:-No}"
    echo "Config:     $DEFAULT_CONFIG"
    echo "COLMAP:     $DEFAULT_COLMAP_CONFIG"
    echo "Quality:    PSNR>=25 SSIM>=0.80 LPIPS<=0.18"
    echo "Log:        $MONITOR_LOG"
    echo "=============================================="

    # Create tmux session
    tmux new-session -d -s "$SESSION_NAME" -n training -c "$PROJECT_DIR"
    tmux send-keys -t "${SESSION_NAME}:training" "cd $PROJECT_DIR && docker compose exec -T edgs-app bash -c \"$DOCKER_CMD\" 2>&1 | tee '$MONITOR_LOG'" Enter

    # Status window
    tmux new-window -t "$SESSION_NAME" -n status -c "$PROJECT_DIR"
    tmux send-keys -t "${SESSION_NAME}:status" "watch -n 30 'echo \"=== Batch Progress ===\"; grep -E \"ATTEMPT|SUCCESS|SKIP|FAIL:|EXHAUSTED|Quality insufficient\" $MONITOR_LOG 2>/dev/null | tail -15; echo; echo \"=== GPU ===\"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null'" Enter

else
    # --- Single video mode ---
    SCENE_NAME=$(basename "$OUTPUT_PATH")
    MONITOR_LOG="${PROJECT_DIR}/outputs/logs/${SCENE_NAME}_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$MONITOR_LOG")"

    DOCKER_INPUT="/EDGS/data/$(echo "$INPUT_VIDEO" | sed 's|.*/data/||')"
    DOCKER_OUTPUT="/EDGS/outputs/$(echo "$OUTPUT_PATH" | sed 's|.*/outputs/||')"

    MONITOR_CMD="bash script/edgs_monitor.sh '$DOCKER_INPUT' '$DOCKER_OUTPUT'"
    [ -n "$FLAG_360" ] && MONITOR_CMD="$MONITOR_CMD --360"
    MONITOR_CMD="$MONITOR_CMD --colmap-config '$DEFAULT_COLMAP_CONFIG'"
    MONITOR_CMD="$MONITOR_CMD --train-config '$DEFAULT_CONFIG'"
    MONITOR_CMD="$MONITOR_CMD --max-retries $MAX_RETRIES"

    echo "=============================================="
    echo "EDGS Training in Tmux (Auto-retry)"
    echo "=============================================="
    echo "Session:    $SESSION_NAME"
    echo "Input:      $INPUT_VIDEO"
    echo "Output:     $OUTPUT_PATH"
    echo "360 mode:   ${FLAG_360:-No}"
    echo "Config:     $DEFAULT_CONFIG"
    echo "COLMAP:     $DEFAULT_COLMAP_CONFIG"
    echo "Quality:    PSNR>=25 SSIM>=0.80 LPIPS<=0.18"
    echo "Log:        $MONITOR_LOG"
    echo "=============================================="

    # Create tmux session
    tmux new-session -d -s "$SESSION_NAME" -n training -c "$PROJECT_DIR"
    tmux send-keys -t "${SESSION_NAME}:training" "cd $PROJECT_DIR && docker compose exec -T edgs-app bash -c \"$MONITOR_CMD\" 2>&1 | tee '$MONITOR_LOG'" Enter

    # Status window
    tmux new-window -t "$SESSION_NAME" -n status -c "$PROJECT_DIR"
    tmux send-keys -t "${SESSION_NAME}:status" "watch -n 30 'echo \"=== Quality Monitor ===\"; grep -E \"ATTEMPT|PSNR|SUCCESS|FAIL:|Diagnosis|deficit|Gaussians\" $MONITOR_LOG 2>/dev/null | tail -15; echo; echo \"=== GPU ===\"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null'" Enter
fi

tmux select-window -t "${SESSION_NAME}:training"

echo ""
echo "Started in tmux session: $SESSION_NAME"
echo ""
echo "  Attach:   tmux attach -t $SESSION_NAME"
echo "  Detach:   Ctrl+b, then d"
echo "  Windows:  Ctrl+b, then n/p"
echo "  Kill:     tmux kill-session -t $SESSION_NAME"
echo ""
echo "You can safely close this terminal."
echo "=============================================="
