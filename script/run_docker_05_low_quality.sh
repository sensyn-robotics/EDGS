#!/bin/bash
# Low memory Docker script for EDGS training - conservative settings for 8-12GB GPUs

echo "🛡️ EDGS Docker Training - Low Quality Mode"
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

# Extreme memory settings
MEMORY_OPTS="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 TORCH_USE_CUDA_DSA=1"

# Clear GPU memory aggressively
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

# Using pre-configured low memory settings

echo -e "\n🎯 Low Quality Configuration:"
echo "  - Config: train_05_low_quality (memory-safe, extended training)"
echo "  - Batch size: 1"
echo "  - Image size: 256 (aggressive reduction for RTX 4080)"
echo "  - Memory caching: DISABLED"
echo "  - Memory optimizations: AGGRESSIVE"
echo ""

# Run with low memory settings
echo "🏃 Starting training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_latter_improved \
        --config train_05_low_quality \
        --max_image_size 512
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Cleaning up..."
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
    echo "📁 Results saved to: outputs/tower_latter/"
else
    echo -e "\n❌ Training failed with exit code: $EXIT_CODE"
    echo "💡 If you still get CUDA OOM, try: script/run_docker_ultra_low_memory.sh"
fi

exit $EXIT_CODE