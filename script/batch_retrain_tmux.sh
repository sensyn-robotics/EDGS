#!/bin/bash
# EDGS Batch Retrain with Tmux and Battery Monitoring
# Usage: ./script/batch_retrain_tmux.sh
#
# Features:
# - Runs in tmux (survives terminal/laptop close)
# - Battery monitoring: suspends when < 10%
# - Auto-retry until PSNR > 20 for each scene
# - Status window for quick monitoring

set -e

SESSION_NAME="edgs"
PROJECT_DIR="/home/mas/proj/sensyn/EDGS"
LOG_DIR="${PROJECT_DIR}/outputs/20260121_otowa"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="${LOG_DIR}/batch_retrain_${TIMESTAMP}.log"
BATTERY_LOG="${LOG_DIR}/battery_${TIMESTAMP}.log"

# Kill existing session if running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 2
fi

# Kill any running training processes
pkill -f "batch_retrain_all.sh" 2>/dev/null || true
sleep 1

echo "=============================================="
echo "EDGS Batch Retrain in Tmux"
echo "=============================================="
echo "Session:      $SESSION_NAME"
echo "Batch log:    $BATCH_LOG"
echo "Battery log:  $BATTERY_LOG"
echo "=============================================="

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -n training -c "$PROJECT_DIR"

# Start batch training in the training window
tmux send-keys -t "${SESSION_NAME}:training" "bash script/batch_retrain_all.sh 2>&1 | tee $BATCH_LOG" Enter

# Wait for training to start
sleep 2

# Create battery monitor window
tmux new-window -t "$SESSION_NAME" -n battery -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:battery" "bash script/battery_monitor.sh 2>&1 | tee $BATTERY_LOG" Enter

# Create status window
tmux new-window -t "$SESSION_NAME" -n status -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:status" "watch -n 30 'echo \"=== Batch Progress ===\"; tail -10 $BATCH_LOG 2>/dev/null; echo; echo \"=== Current Training ===\"; ls -t ${LOG_DIR}/*_attempt*.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null | grep -E \"ITER|PSNR\" | tail -2; echo; echo \"=== GPU ===\"; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null; echo; echo \"=== Battery ===\"; cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1 | xargs -I{} echo \"{}%\" || echo \"N/A\"'" Enter

# Create log window for live training output
tmux new-window -t "$SESSION_NAME" -n logs -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:logs" "tail -f $BATCH_LOG" Enter

# Select training window
tmux select-window -t "${SESSION_NAME}:training"

echo ""
echo "Batch training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+b, then d"
echo "  Switch windows:      Ctrl+b, then n (next) or p (previous)"
echo "  Kill session:        tmux kill-session -t $SESSION_NAME"
echo ""
echo "Windows:"
echo "  0: training - Main batch training process"
echo "  1: battery  - Battery monitor (auto-suspend at <10%)"
echo "  2: status   - Quick status view (updates every 30s)"
echo "  3: logs     - Live batch log"
echo ""
echo "Training will:"
echo "  - Continue even if you close the terminal/laptop"
echo "  - Auto-suspend when battery < 10%"
echo "  - Auto-retry each scene until PSNR > 20"
echo ""
echo "You can safely close this terminal now."
echo "=============================================="
