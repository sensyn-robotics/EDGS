#!/bin/bash
# Medium quality Docker script for EDGS training - good quality with memory efficiency for 8-12GB GPUs

echo "⭐ EDGS Docker Training - Medium Quality Mode ⭐⭐⭐"
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

# Balanced memory settings
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

echo -e "\n⭐ Medium Quality Configuration:"
echo "  - Config: train_04_medium_quality (balanced approach)"
echo "  - Batch size: 1 (memory safe)"
echo "  - SH degree: 1 (memory efficient)"
echo "  - Training epochs: 20,000 (reasonable time)"
echo "  - Limited densification: ENABLED"
echo "  - EDGS correlation: ENABLED (reduced)"
echo "  - Image size: 1024px (good resolution)"
echo "  - Training time: ~1-1.5 hours"
echo "  - Expected PSNR: 16-20"
echo ""
echo "🚀 Good compromise between quality and memory efficiency!"
echo ""

# Run with medium quality settings
echo "🏃 Starting medium quality training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_latter_medium_quality \
        --config train_04_medium_quality \
        --max_image_size 1024
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ Medium quality training completed successfully!"
    echo "📁 Results saved to: outputs/tower_latter_medium_quality/"
    echo "🎯 Balanced approach achieved good quality with reasonable training time!"
else
    echo -e "\n❌ Training failed with exit code: $EXIT_CODE"
    echo "💡 If you get CUDA OOM, try run_docker_05_low_quality.sh for safer settings"
    echo "💡 Alternative: try run_docker_06_lowest_quality.sh for 4-8GB GPUs"
fi

exit $EXIT_CODE