#!/usr/bin/env python
# coding: utf-8

# # EDGS: Eliminating Densification for Gaussian Splatting
# EDGS improves 3D Gaussian Splatting by removing the need for densification. It starts from a dense point cloud initialization based on 2D correspondences, leading to:
# - ⚡ Faster convergence (only 25% of training time)
#  - 🌀 Higher rendering quality
#  - 💡 No need for progressive densification

# ## 2. Import libraries
import argparse
import logging
import os
import random
import sys
import shutil

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from hydra import compose, initialize
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

# Add the project root directory to sys.path
# so that modules from 'source' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# sys.path.append("../submodules/gaussian-splatting")
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
from source.utils_preprocess import (
    orchestrate_video_to_colmap_scene,  # Use the refactored function
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def copy_config_to_output(config_name, model_path):
    """Copy the config YAML file to the output directory for reference."""
    config_file = os.path.join(project_root, "configs", f"{config_name}.yaml")
    if os.path.exists(config_file):
        dest_file = os.path.join(model_path, f"config_{config_name}.yaml")
        shutil.copy2(config_file, dest_file)
        print(f"Config copied to: {dest_file}")
    else:
        print(f"Warning: Config file not found: {config_file}")

# --- Add argument parsing ---
parser = argparse.ArgumentParser(
    description="Fit EDGS model to a scene, optionally from a video."
)
parser.add_argument(
    "--video_path",
    type=str,
    default=os.path.join(
        project_root, "assets", "examples", "video_fruits.mp4"
    ),  # Use project_root
    help="Path to the input video file.",
)
parser.add_argument(
    "--colmap_output_path",
    type=str,
    default=None,
    help="Path to COLMAP scene directory. If it exists, uses it directly. If not, creates COLMAP output here when processing video.",
)
parser.add_argument(
    "--config",
    type=str,
    default="train",
    help="Config name to use (e.g., 'train', 'train_low_memory', 'train_very_low_memory'). Default: train",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Directory to save EDGS training results. If not specified, creates 'models' subfolder in COLMAP scene or colmap_output_path.",
)
parser.add_argument(
    "--target_fps",
    type=float,
    default=3.0,
    help="Target frames per second for video extraction. Higher values extract more frames. Default: 3.0",
)
parser.add_argument(
    "--max_image_size",
    type=int,
    default=-1,
    help="Maximum image dimension (width or height). Images larger than this will be resized while preserving aspect ratio. For example, a 4K image (3840x2160) with max_image_size=1920 becomes 1920x1080. Use -1 to keep original resolution. Default: -1",
)
args = parser.parse_args()
# --- End argument parsing ---

with initialize(config_path="../configs", version_base="1.1"):
    cfg = compose(config_name=args.config)
    print(f"\n📋 Using config: {args.config}")
    if "low_memory" in args.config:
        print("💾 Low memory mode enabled - densification disabled, reduced batch size")

SAME_WITH_GRADIO_DEMO = False
if SAME_WITH_GRADIO_DEMO:
    cfg.gs.opt.opacity_reset_interval = 1_000_000
    cfg.train.reduce_opacity = True
    cfg.train.no_densify = True
    cfg.train.max_lr = True
    cfg.train.gs_epochs = 1000

    cfg.init_wC.use = False  # Disable for fallback cases
    cfg.init_wC.nns_per_ref = 1
    cfg.init_wC.add_SfM_init = False
    cfg.init_wC.scaling_factor = 0.00077 * 2.0
    cfg.init_wC.num_refs = 10  # Use more reference views for better reconstruction
    cfg.init_wC.matches_per_ref = 20000

# Print memory usage info if in low memory mode
if "low_memory" in args.config:
    print(f"\n🔧 Memory-saving settings:")
    print(f"  - Batch size: {cfg.gs.opt.batch_size}")
    print(f"  - Densification: {'Disabled' if cfg.train.no_densify else 'Enabled'}")
    print(f"  - Reference images: {cfg.init_wC.num_refs}")
    print(f"  - Matches per ref: {cfg.init_wC.matches_per_ref}")
    
print("\n" + "="*50)
print(OmegaConf.to_yaml(cfg))


# # 3. Init input parameters

# ## 3.1 Set up scene directory
# Check if colmap_output_path exists and is a valid COLMAP scene
use_existing_colmap = False
if args.colmap_output_path and os.path.exists(args.colmap_output_path):
    # Check if it's a valid COLMAP scene
    sparse_dir = os.path.join(args.colmap_output_path, "sparse", "0")
    if os.path.exists(sparse_dir):
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(sparse_dir, f))]
        if not missing_files:
            use_existing_colmap = True
            scene_dir = args.colmap_output_path
            print(f"Using existing COLMAP scene: {scene_dir}")

if not use_existing_colmap:
    # Process video to create COLMAP scene
    if not os.path.exists(args.video_path):
        print(f"Error: Video file does not exist: {args.video_path}")
        sys.exit(1)
    
    # Set default colmap_output_path if not specified
    if not args.colmap_output_path:
        args.colmap_output_path = os.path.join(project_root, "outputs")
        
    print(f"Starting video processing for: {args.video_path}")
    try:
        # The first return value 'images_data' might not be directly used by the trainer
        # if the Scene object loads everything from the COLMAP directory.
        # Use original image size if max_image_size is -1
        max_size = args.max_image_size if args.max_image_size > 0 else 999999
        
        _, scene_dir = orchestrate_video_to_colmap_scene(
            args.video_path,
            cfg.init_wC.num_refs,  # Assuming you added this arg
            max_size=max_size,  # Use configurable size
            base_work_dir=args.colmap_output_path,  # Assuming you added this arg
            use_automatic_mode=True,  # Use automatic reconstructor-like settings
            target_fps=args.target_fps,  # Pass target FPS for frame extraction
        )
        if scene_dir is None:
            print(f"Failed to process video {args.video_path}. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during video preprocessing: {e}")
        sys.exit(1)

# Set up paths for EDGS
cfg.gs.dataset.source_path = scene_dir

# Determine output directory for EDGS training results
if args.output_path:
    # Use specified output directory
    model_path = args.output_path
else:
    # Default behavior: create models subfolder in scene directory
    model_path = os.path.join(scene_dir, "models")

cfg.gs.dataset.model_path = model_path
print(f"COLMAP scene: {scene_dir}")
print(f"EDGS output: {model_path}")
os.makedirs(model_path, exist_ok=True)

# Copy config file to output directory for reference
copy_config_to_output(args.config, model_path)


# # 4. Initilize model and logger
if cfg.wandb.mode != "disabled":
    logging.info(
        "wandb logging is enabled (mode={}). Results will be logged to wandb.".format(
            cfg.wandb.mode
        )
    )
    _ = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
    )
else:
    logging.info(
        "wandb logging is disabled (mode={}). Results will not be logged to wandb.".format(
            cfg.wandb.mode
        )
    )
omegaconf.OmegaConf.resolve(cfg)
set_seed(cfg.seed)
# Init output folder
print("Output folder: {}".format(cfg.gs.dataset.model_path))
os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
# Init gs model
gs = hydra.utils.instantiate(cfg.gs)
trainer = EDGSTrainer(
    GS=gs,
    training_config=cfg.gs.opt,
    device=cfg.device,
    log_wandb=(cfg.wandb.mode != "disabled"),
)


# # 5. Init with matchings
trainer.timer.start()
trainer.init_with_corr(cfg.init_wC)
trainer.timer.pause()


# ### Visualize a few initial viewpoints
with torch.no_grad():
    viewpoint_stack = trainer.GS.scene.getTrainCameras()
    available_cams = trainer.GS.scene.getTrainCameras()
    num_cams_to_viz = min(4, len(available_cams))
    viewpoint_cams_to_viz = random.sample(available_cams, num_cams_to_viz)
    for idx, viewpoint_cam in enumerate(viewpoint_cams_to_viz):
        render_pkg = trainer.GS(viewpoint_cam)
        image = render_pkg["render"]

        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
        image_gt_np = (
            viewpoint_cam.original_image.clone()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(image_gt_np)
        ax[0].axis("off")
        ax[1].imshow(image_np)
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cfg.gs.dataset.model_path,
                f"viewpoint_{idx}_initial.png",
            )
        )
        plt.show()
        plt.close(fig)


# # 6.Optimize scene
# Optimize first briefly for 5k steps and visualize results. We also disable saving of pretrained models. Train function can be changed for any other method
trainer.saving_iterations = []
# cfg.train.gs_epochs = 5_000
trainer.train(cfg.train)


# ### Visualize same viewpoints
with torch.no_grad():
    for viewpoint_cam in viewpoint_cams_to_viz:
        render_pkg = trainer.GS(viewpoint_cam)
        image = render_pkg["render"]

        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
        image_gt_np = (
            viewpoint_cam.original_image.clone()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(image_gt_np)
        ax[0].axis("off")
        ax[1].imshow(image_np)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()


# ### Save model
with torch.no_grad():
    trainer.save_model()


# # # 7. Continue training until we reach total 30K training steps
# cfg.train.gs_epochs = 25_000
# trainer.train(cfg.train)


# # ### Visualize same viewpoints
# with torch.no_grad():
#     for viewpoint_cam in viewpoint_cams_to_viz:
#         render_pkg = trainer.GS(viewpoint_cam)
#         image = render_pkg["render"]

#         image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
#         image_gt_np = (
#             viewpoint_cam.original_image.clone()
#             .detach()
#             .cpu()
#             .numpy()
#             .transpose(1, 2, 0)
#         )

#         # Clip values to be in the range [0, 1]
#         image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
#         image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#         ax[0].imshow(image_gt_np)
#         ax[0].axis("off")
#         ax[1].imshow(image_np)
#         ax[1].axis("off")
#         plt.tight_layout()
#         plt.show()


# ### Save model
# with torch.no_grad():
#     trainer.save_model()
