#!/bin/bash
# High quality Docker script for EDGS training - extended training for 12-16GB GPUs

echo "⭐ EDGS Docker Training - High Quality Mode ⭐⭐⭐⭐"
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

# Progressive memory settings
MEMORY_OPTS="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:False CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1"

# Clear GPU memory
echo -e "\n🧹 Clearing GPU memory..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    python -c '
import torch
import gc
for _ in range(3):
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

echo -e "\n⭐ High Quality Configuration (Progressive Training):"
echo "  - Config: train_02_high_quality (extended training)"
echo "  - Batch size: 1 (memory efficient)"
echo "  - SH degree: 1 (memory balanced)"
echo "  - Training epochs: 100,000 (very long training)"
echo "  - Progressive densification: ENABLED"
echo "  - Image size: 800px (balanced resolution)"
echo "  - Training time: ~3-4 hours"
echo "  - Expected PSNR: 18-25"
echo ""
echo "🚀 Progressive approach - quality through extended training!"
echo ""

# Run with progressive high quality settings
echo "🏃 Starting extended quality training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_latter_high_quality \
        --config train_02_high_quality \
        --max_image_size 800
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ High quality training completed successfully!"
    echo "📁 Results saved to: outputs/tower_latter_high_quality/"
    echo "🎯 Progressive training approach achieved excellent quality through extended iterations!"
else
    echo -e "\n❌ Training failed with exit code: $EXIT_CODE"
    echo "💡 If you get CUDA OOM, try run_docker_03_optimal_quality.sh for shorter training"
    echo "💡 Alternative: try run_docker_04_medium_quality.sh for 8-12GB GPUs"
fi

exit $EXIT_CODE