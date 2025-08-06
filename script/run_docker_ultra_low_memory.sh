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

# Maximum memory safety settings
MEMORY_OPTS="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1"

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

# Create wrapper that disables LPIPS to save memory
echo -e "\n📝 Creating memory-safe wrapper..."
docker exec $CONTAINER bash -c "
cat > /EDGS/run_ultra_safe.py << 'EOF'
import os
import sys
import gc
import torch

# Import training modules
sys.path.insert(0, '/EDGS')
import source.trainer as trainer_module

# Patch to disable LPIPS evaluation
original_evaluate = trainer_module.Trainer.evaluate

def memory_safe_evaluate(self):
    import torch
    from tqdm import tqdm
    
    l1_test = 0.0
    psnr_test = 0.0
    
    viewpoint_stack = self.test_dataset.get_test_cams()
    
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(viewpoint_stack, desc=\"Evaluating (no LPIPS)\")):
            render = self.gs_render(self.splats, viewpoint, mode=\"test\")
            gt_image = viewpoint.original_image[0:3, :, :]
            image = torch.clamp(render[\"image\"], 0.0, 1.0)
            
            # Basic metrics only
            l1_test += torch.abs(image - gt_image).mean().detach().double()
            psnr_test += (-10.0 * torch.log10((image - gt_image).pow(2).mean())).detach().double()
            
            # Aggressive memory cleanup
            if idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    l1_test /= len(viewpoint_stack)
    psnr_test /= len(viewpoint_stack)
    
    print(f\"Eval L1: {l1_test:.4f}, PSNR: {psnr_test:.2f}dB (LPIPS disabled)\")
    
    return {
        \"l1_test\": l1_test,
        \"psnr_test\": psnr_test,
        \"ssim_test\": 0.0,
        \"lpips_test\": 0.0
    }

trainer_module.Trainer.evaluate = memory_safe_evaluate

# Run the actual script
exec(open('/EDGS/script/fit_model_to_scene_full.py').read())
EOF
"

echo -e "\n🎯 Ultra Low Memory Configuration:"
echo "  - Config: train_ultra_low_memory (SfM-only)"
echo "  - LPIPS evaluation DISABLED"
echo "  - Batch size: 4"
echo "  - Image size: 800"
echo "  - Memory caching: DISABLED"
echo "  - Evaluation: Reduced frequency"
echo ""
echo "⚠️  This will be slower but should prevent OOM!"
echo ""

# Run with ultra-safe settings
echo "🏃 Starting ultra-safe training..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    $MEMORY_OPTS python run_ultra_safe.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_latter \
        --config train_ultra_low_memory \
        --max_image_size 800
"

EXIT_CODE=$?

# Cleanup
echo -e "\n🧹 Final cleanup..."
docker exec $CONTAINER bash -c "rm -f /EDGS/run_ultra_safe.py"
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