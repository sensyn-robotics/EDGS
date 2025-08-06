#!/bin/bash
# Ultra low memory Docker script - absolute minimal settings to avoid system freeze

echo "🆘 EDGS Docker Training - Ultra Low Memory Mode"
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

# Ultimate memory safety settings
MEMORY_OPTS="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:False CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 TORCH_USE_CUDA_DSA=1"

# Aggressive GPU memory cleanup
echo -e "\n🧹 Aggressively clearing GPU memory..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    python -c '
import torch
import gc
import os

# Force cleanup
os.system(\"pkill -f python || true\")
for _ in range(5):
    gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if hasattr(torch.cuda, \"reset_peak_memory_stats\"):
        torch.cuda.reset_peak_memory_stats()
    print(\"GPU memory aggressively cleared\")
'
"

# Check memory status
echo -e "\n📊 GPU memory status:"
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python /EDGS/script/check_gpu_memory.py"

# Using pre-configured ultra low memory settings

echo -e "\n🎯 Ultra Low Memory Configuration:"
echo "  - Config: train_ultra_low_memory (SfM-only)"
echo "  - Batch size: 1 (ultimate minimum)"
echo "  - Image size: 128 (ultra tiny for extreme memory savings)"
echo "  - Memory caching: DISABLED"
echo "  - Evaluation: Every 7500 iterations (reduced frequency)"
echo ""
echo "⚠️  This will be slower but should prevent OOM!"
echo ""

# Run with ultra-safe settings directly
echo "🏃 Starting ultra-safe training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter_minimal_final \
        --output_path outputs/tower_latter_ultra_low_memory \
        --config train_ultra_low_memory \
        --max_image_size 64
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Final cleanup..."
docker exec $CONTAINER bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✅ Training completed successfully!"
    echo "📁 Results saved to: outputs/tower_latter/"
    echo ""
    echo "📌 Ultra-safe mode used:"
    echo "  - SfM initialization only"
    echo "  - LPIPS disabled"
    echo "  - Minimal batch size and image resolution"
else
    echo -e "\n❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "🆘 Last resort options:"
    echo "  1. Close ALL other applications"
    echo "  2. docker compose down && docker compose up -d"
    echo "  3. Use a GPU with more memory (16GB+)"
fi

exit $EXIT_CODE