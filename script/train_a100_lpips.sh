#!/bin/bash
# EDGS Training Script Optimized for NVIDIA A100 GPU
# Target: LPIPS < 0.2

echo "========================================="
echo "EDGS A100-Optimized Training for LPIPS < 0.2"
echo "========================================="
echo "This configuration leverages A100's 40GB memory"
echo ""

COLMAP_PATH="${1:-outputs/closeup_flight_all/video_scene}"
OUTPUT_PATH="${2:-outputs/closeup_flight_all_a100_lpips}"

echo "Input COLMAP path: $COLMAP_PATH"
echo "Output path: $OUTPUT_PATH"
echo ""

python script/train.py \
  --config-name=train_01_highest_quality \
  train.gs_epochs=200000 \
  train.no_densify=False \
  gs.dataset.source_path=$COLMAP_PATH \
  gs.dataset.model_path=$OUTPUT_PATH \
  init_wC.use=True \
  init_wC.matches_per_ref=50000 \
  init_wC.nns_per_ref=15 \
  init_wC.num_refs=500 \
  init_wC.scaling_factor=0.0005 \
  init_wC.proj_err_tolerance=0.005 \
  init_wC.roma_model="outdoors" \
  init_wC.add_SfM_init=True \
  gs.opt.batch_size=8 \
  gs.dataset.resolution=1 \
  gs.opt.lambda_dssim=0.6 \
  gs.sh_degree=3 \
  gs.opt.densification_interval=250 \
  gs.opt.densify_until_iter=100000 \
  gs.opt.densify_grad_threshold=0.00008 \
  gs.opt.opacity_reset_interval=2000 \
  gs.opt.percent_dense=0.01 \
  gs.opt.position_lr_init=0.00016 \
  gs.opt.position_lr_final=0.0000001 \
  gs.opt.position_lr_delay_mult=0.005 \
  gs.opt.position_lr_max_steps=200000 \
  gs.opt.feature_lr=0.008 \
  gs.opt.opacity_lr=0.04 \
  gs.opt.scaling_lr=0.004 \
  gs.opt.rotation_lr=0.0008 \
  gs.opt.exposure_lr_init=0.008 \
  gs.opt.exposure_lr_final=0.00008 \
  gs.opt.save_iterations="[10000, 20000, 30000, 40000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]" \
  wandb.mode=disabled \
  wandb.project="edgs_a100_lpips"

echo ""
echo "Training started. Monitor progress in the output folder."
echo "Target: Test LPIPS < 0.2"
echo "Expected training time on A100: 3-5 hours"