#!/bin/bash
# Batch EDGS training for all 360 videos with automatic quality retry
# Usage: ./script/batch_train_360.sh <data_dir> <output_base_dir>
#
# Runs inside Docker container. Processes each video with edgs_monitor.sh which
# retries with adjusted parameters until quality thresholds are met:
#   PSNR >= 25.0, SSIM >= 0.80, LPIPS <= 0.18
#
# Example:
#   docker compose exec edgs-app bash -c "./script/batch_train_360.sh /EDGS/data/20260121_otowa /EDGS/outputs/20260121_otowa"

set -e

DATA_DIR="${1:?Usage: $0 <data_dir> <output_base_dir>}"
OUTPUT_BASE="${2:?Usage: $0 <data_dir> <output_base_dir>}"

COLMAP_CONFIG="colmap_08_360_2fps"
TRAIN_CONFIG="train_large_lowmem"
MIN_PSNR=25.0
MIN_SSIM=0.80
MAX_LPIPS=0.18
MAX_RETRIES=10

echo "=============================================="
echo "EDGS Batch Training - 360 Video (2fps)"
echo "=============================================="
echo "Data directory:  ${DATA_DIR}"
echo "Output base:     ${OUTPUT_BASE}"
echo "COLMAP config:   ${COLMAP_CONFIG}"
echo "Training config: ${TRAIN_CONFIG}"
echo "Quality:         PSNR>=${MIN_PSNR} SSIM>=${MIN_SSIM} LPIPS<=${MAX_LPIPS}"
echo "=============================================="

# Collect all video files
VIDEO_FILES=()
for ext in mp4 mov avi MP4 MOV AVI; do
    while IFS= read -r -d '' f; do
        VIDEO_FILES+=("$f")
    done < <(find "${DATA_DIR}" -maxdepth 1 -name "*.${ext}" -print0 2>/dev/null)
done

# Sort for consistent ordering
IFS=$'\n' VIDEO_FILES=($(sort <<<"${VIDEO_FILES[*]}")); unset IFS

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    echo "ERROR: No video files found in ${DATA_DIR}"
    exit 1
fi

echo ""
echo "Found ${#VIDEO_FILES[@]} videos:"
for v in "${VIDEO_FILES[@]}"; do
    echo "  - $(basename "$v")"
done
echo ""

TOTAL=${#VIDEO_FILES[@]}
SUCCESS=0
FAILED=0
SKIPPED=0

for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO")
    # Create output dir name: strip extension, take prefix before _VID
    OUTPUT_NAME=$(echo "$BASENAME" | sed 's/\.[^.]*$//' | sed 's/_VID_.*//')
    OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_NAME}"

    echo ""
    echo "=============================================="
    echo "[$(($i + 1))/${TOTAL}] ${BASENAME}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "=============================================="

    # Check if already completed with good quality (check both last and best logs)
    for CHECK_LOG in "${OUTPUT_DIR}_train.log" "${OUTPUT_DIR}_best_train.log"; do
        if [ -f "$CHECK_LOG" ]; then
            last_test=$(grep -E "Evaluating test:.*PSNR=" "$CHECK_LOG" 2>/dev/null | tail -1) || true
            if [ -n "$last_test" ]; then
                psnr=$(echo "$last_test" | grep -oP 'PSNR=\K[0-9.]+') || psnr="0"
                ssim=$(echo "$last_test" | grep -oP 'SSIM=\K[0-9.]+') || ssim="0"
                lpips=$(echo "$last_test" | grep -oP 'LPIPS_splat=\K[0-9.]+') || lpips="1"
                pass=$(python3 -c "
p = float('$psnr') >= $MIN_PSNR
s = float('$ssim') >= $MIN_SSIM
l = float('$lpips') <= $MAX_LPIPS
print(1 if p and s and l else 0)
")
                if [ "$pass" = "1" ]; then
                    echo "  SKIP: Already meets quality (PSNR=${psnr} SSIM=${ssim} LPIPS=${lpips}) [$(basename "$CHECK_LOG")]"
                    SKIPPED=$((SKIPPED + 1))
                    continue 2
                fi
            fi
        fi
    done

    START_TIME=$(date +%s)

    if bash script/edgs_monitor.sh "$VIDEO" "$OUTPUT_DIR" \
        --360 \
        --colmap-config "$COLMAP_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --min-psnr "$MIN_PSNR" \
        --min-ssim "$MIN_SSIM" \
        --max-lpips "$MAX_LPIPS" \
        --max-retries "$MAX_RETRIES"; then

        END_TIME=$(date +%s)
        ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
        echo ""
        echo "  DONE: ${BASENAME} completed in ${ELAPSED} minutes"
        SUCCESS=$((SUCCESS + 1))
    else
        END_TIME=$(date +%s)
        ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
        echo ""
        echo "  FAILED: ${BASENAME} after ${ELAPSED} minutes (exhausted retries)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "Batch Training Complete"
echo "=============================================="
echo "Total:   ${TOTAL}"
echo "Success: ${SUCCESS}"
echo "Failed:  ${FAILED}"
echo "Skipped: ${SKIPPED}"
echo "=============================================="
