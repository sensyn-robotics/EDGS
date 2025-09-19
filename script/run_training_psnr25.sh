#!/bin/bash
# Direct EDGS training script for PSNR > 25

echo "🚀 EDGS Training for Steel Tower - Target PSNR > 25"
echo "=============================================================="

# Step 1: Fix NumPy version
echo "1️⃣ Fixing NumPy compatibility..."
pip install -q "numpy==1.26.4"

# Step 2: Set CUDA environment (will work with or without CUDA)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'

# Step 3: Check if CUDA works
echo "2️⃣ Checking CUDA availability..."
python -c "
import warnings
warnings.filterwarnings('ignore')
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA not available - will use CPU (slower)')
except Exception as e:
    print(f'⚠️ {e}')
"

# Step 4: Run training
echo ""
echo "3️⃣ Starting EDGS training..."
echo "=============================================================="

COLMAP_PATH="outputs/closeup_flight_275/video_scene"
OUTPUT_DIR="outputs/closeup_flight_275/edgs_psnr25"
LOG_FILE="outputs/closeup_flight_275/training_psnr25.log"

if [ ! -d "$COLMAP_PATH" ]; then
    echo "❌ COLMAP path not found: $COLMAP_PATH"
    echo "Available directories:"
    ls -la outputs/closeup_flight_275/
    exit 1
fi

echo "📁 Input: $COLMAP_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo ""

mkdir -p $OUTPUT_DIR

# Run training with optimized parameters for PSNR > 25
echo "Running training (this may take 30-60 minutes)..."
echo ""

python script/train.py \
    train.gs_epochs=35000 \
    train.no_densify=True \
    gs.dataset.source_path=$COLMAP_PATH \
    gs.dataset.model_path=$OUTPUT_DIR \
    init_wC.matches_per_ref=22000 \
    init_wC.nns_per_ref=4 \
    init_wC.num_refs=200 \
    gs.opt.position_lr_init=0.00018 \
    gs.opt.position_lr_final=0.0000018 \
    gs.opt.position_lr_max_steps=35000 \
    gs.opt.feature_lr=0.0028 \
    gs.opt.opacity_lr=0.11 \
    gs.opt.scaling_lr=0.0055 \
    gs.opt.rotation_lr=0.0023 \
    gs.opt.percent_dense=0.002 \
    gs.opt.densification_interval=500 \
    gs.opt.opacity_reset_interval=3000 \
    gs.opt.densify_from_iter=500 \
    gs.opt.densify_until_iter=15000 \
    2>&1 | tee $LOG_FILE

# Step 5: Check results
echo ""
echo "4️⃣ Checking results..."
echo "=============================================================="

if [ -f "$LOG_FILE" ]; then
    # Extract PSNR values
    echo "PSNR progression during training:"
    grep -E "Iter.*PSNR" $LOG_FILE | tail -10

    # Get final PSNR
    FINAL_PSNR=$(grep -oP "PSNR.*: \K[0-9.]+" $LOG_FILE | tail -1)

    if [ ! -z "$FINAL_PSNR" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Final PSNR: $FINAL_PSNR"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if (( $(echo "$FINAL_PSNR >= 25" | bc -l) )); then
            echo ""
            echo "🎉 SUCCESS! Target PSNR > 25 achieved!"
            echo "📁 Trained model saved to: $OUTPUT_DIR"
            echo ""
            echo "Next steps:"
            echo "  - View the model: python script/viewer.py --model_path $OUTPUT_DIR"
            echo "  - Render video: python script/render.py --model_path $OUTPUT_DIR"
        else
            echo ""
            echo "⚠️  PSNR $FINAL_PSNR is below target (25)"
            echo ""
            echo "To improve, you can:"
            echo "  1. Run more training epochs"
            echo "  2. Increase matches_per_ref parameter"
            echo "  3. Use higher quality COLMAP reconstruction"
        fi
    else
        echo "⚠️  Could not extract PSNR from log"
    fi
else
    echo "❌ Training log not found"
fi

echo ""
echo "✅ Training script complete!"