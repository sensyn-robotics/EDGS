#!/bin/bash
# Training script for steel tower EDGS

echo "🚀 Starting EDGS training for Steel Tower"
echo "=============================================================="

# Clear any existing CUDA environment variables
unset CUDA_VISIBLE_DEVICES
unset CUDA_LAUNCH_BLOCKING

# Ensure conda environment is activated
source /opt/conda/etc/profile.d/conda.sh
conda activate edgs

# Clear GPU memory
python -c "
import torch
import gc
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print('GPU memory cleared successfully')
        print(f'CUDA available: {torch.cuda.is_available()}')
        print(f'GPU count: {torch.cuda.device_count()}')
except Exception as e:
    print(f'GPU init error: {e}')
    print('Attempting to continue...')
"

echo -e "\n🎯 Configuration:"
echo "  - Config: train_steel_tower"
echo "  - Source: outputs/closeup_flight_275/video_scene"
echo "  - Output: outputs/closeup_flight_275/edgs_run1"
echo ""

# Run training - remove device override to use CUDA
sed -i '/device: "cpu"/d' /EDGS/configs/train_steel_tower.yaml

# Run the training
cd /EDGS
python script/train.py \
    --config-name=train_steel_tower \
    2>&1 | tee outputs/closeup_flight_275/training_steel_tower.log

echo "Training completed or failed - check log file"