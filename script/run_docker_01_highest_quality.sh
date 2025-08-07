#!/bin/bash
# Highest quality Docker script for EDGS training - best possible results for 16GB+ GPUs

echo "⭐ EDGS Docker Training - Highest Quality Mode ⭐⭐⭐⭐⭐"
echo "=============================================================="

# Check if docker compose is running
if ! docker compose ps | grep -q "edgs-app.*running"; then
    echo "⚠️  Docker container not running. Starting it now..."
    docker compose up -d
    echo "⏳ Waiting for container to be ready..."
    sleep 5
fi

# Get container name
CONTAINER=$(docker compose ps -q edgs-app)
if [ -z "$CONTAINER" ]; then
    echo "❌ Error: Could not find running container"
    exit 1
fi

echo "✅ Found container: $CONTAINER"

# Optimal memory settings for high-end GPUs
MEMORY_OPTS="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True CUDA_LAUNCH_BLOCKING=0 PYTORCH_NO_CUDA_MEMORY_CACHING=0"

# Clear GPU memory
echo -e "\n🧹 Clearing GPU memory..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    python -c '
import torch
import gc
for _ in range(2):
    gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(\"GPU memory cleared\")
'
"

# Check memory status
echo -e "\n📊 GPU memory status:"
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python /EDGS/script/check_gpu_memory.py"

echo -e "\n⭐ Highest Quality Configuration:"
echo "  - Config: train_01_highest_quality (best possible quality)"
echo "  - Batch size: 64 (optimal for convergence)"
echo "  - SH degree: 3 (full spherical harmonics)"
echo "  - EDGS correlation: ENABLED"
echo "  - Image size: 1920px (high resolution)"
echo "  - Training time: ~3-4 hours"
echo "  - Expected PSNR: 25-35+"
echo ""
echo "🚀 This configuration requires 16GB+ GPU memory!"
echo ""

# Run with highest quality settings
echo "🏃 Starting highest quality training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_latter_highest_quality \
        --config train_01_highest_quality \
        --max_image_size 1920
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ Highest quality training completed successfully!"
    echo "📁 Results saved to: outputs/tower_latter_highest_quality/"
    echo "🎯 This should achieve the best possible PSNR results!"
else
    echo -e "\n❌ Training failed with exit code: $EXIT_CODE"
    echo "💡 If you get CUDA OOM, you need a GPU with more memory (RTX 4090, A100, etc.)"
    echo "💡 Alternative: try run_docker_02_high_quality.sh for 12-16GB GPUs"
fi

exit $EXIT_CODE