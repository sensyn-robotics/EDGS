#!/bin/bash
# Progressive multi-resolution training for highest quality results
# Stage 1: Train at low resolution for stable base
# Stage 2: Fine-tune at higher resolution for details

echo "🎯 Progressive Multi-Resolution Training"
echo "=========================================="

# Check if docker compose is running
if ! docker compose ps | grep -q "edgs-app.*running"; then
    echo "⚠️  Docker container not running. Starting it now..."
    docker compose up -d
    echo "⏳ Waiting for container to be ready..."
    sleep 5
fi

CONTAINER=$(docker compose ps -q edgs-app)
if [ -z "$CONTAINER" ]; then
    echo "❌ Error: Could not find running container"
    exit 1
fi

# Stage 1: Low resolution base training (40k iterations)
echo "🏃 Stage 1: Base training at 400px resolution..."
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_progressive_stage1 \
        --config train_progressive_quality \
        --max_image_size 400
"

STAGE1_EXIT=$?
if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 failed with exit code: $STAGE1_EXIT"
    exit $STAGE1_EXIT
fi

# Stage 2: Fine-tune with higher resolution (additional 30k iterations)
echo "🏃 Stage 2: Fine-tuning at 600px resolution..."

# Create fine-tuning config that loads from Stage 1
docker exec $CONTAINER bash -c "
    source /opt/conda/etc/profile.d/conda.sh && conda activate edgs && \
    cd /EDGS && \
    # Copy the trained model as starting point
    cp -r outputs/tower_progressive_stage1/point_cloud/iteration_100000 outputs/tower_progressive_stage1/point_cloud/iteration_0
    
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:False python script/fit_model_to_scene_full.py \
        --colmap_output_path outputs/tower_latter \
        --output_path outputs/tower_progressive_final \
        --config train_progressive_quality \
        --max_image_size 600 \
        --gs_source outputs/tower_progressive_stage1/point_cloud/iteration_100000/point_cloud.ply
"

STAGE2_EXIT=$?

if [ $STAGE2_EXIT -eq 0 ]; then
    echo "✅ Progressive training completed successfully!"
    echo "📁 Final results: outputs/tower_progressive_final/"
    echo "🎯 This approach should achieve much higher PSNR through:"
    echo "   - 100k+ total iterations"
    echo "   - Progressive resolution increase"
    echo "   - Extended densification period"
    echo "   - Fine-tuned learning schedules"
else
    echo "❌ Stage 2 failed with exit code: $STAGE2_EXIT"
fi

exit $STAGE2_EXIT