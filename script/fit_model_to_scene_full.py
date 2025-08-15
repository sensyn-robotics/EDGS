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
import cv2
from pathlib import Path

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
    run_colmap_on_scene,  # Direct COLMAP runner
)
from script.extract_frames_uniform import extract_frames_uniformly

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def find_videos_in_directory(directory_path, extensions=('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI')):
    """
    Recursively find all video files in a directory and its subdirectories.
    
    Args:
        directory_path: Root directory to search
        extensions: Tuple of valid video file extensions
        
    Returns:
        List of video paths
    """
    videos = []
    directory_path = Path(directory_path)
    
    for ext in extensions:
        for video_path in directory_path.rglob(f'*{ext}'):
            videos.append(str(video_path))
    
    return sorted(videos)


def copy_config_to_output(config_name, model_path):
    """Copy the config YAML file to the output directory for reference."""
    config_file = os.path.join(project_root, "configs", f"{config_name}.yaml")
    if os.path.exists(config_file):
        dest_file = os.path.join(model_path, f"config_{config_name}.yaml")
        shutil.copy2(config_file, dest_file)
        print(f"Config copied to: {dest_file}")
    else:
        print(f"Warning: Config file not found: {config_file}")


def process_video_to_colmap_scene(video_path, output_path, target_fps=3.0, max_image_size=1024, colmap_config="very_low_memory"):
    """
    Process video(s) with uniform frame extraction across entire duration and run COLMAP.
    Can handle both single video file or directory containing multiple videos.
    
    Args:
        video_path: Path to input video file or directory containing videos
        output_path: Directory to save COLMAP scene
        target_fps: Effective target fps for frame extraction
        max_image_size: Maximum dimension for frames
        colmap_config: COLMAP configuration preset
        
    Returns:
        scene_dir: Path to COLMAP scene directory
    """
    # Check if input is a directory or single video file
    is_directory = os.path.isdir(video_path)
    
    if is_directory:
        print(f"🎥 Processing multiple videos from directory: {video_path}")
        videos = find_videos_in_directory(video_path)
        if not videos:
            raise ValueError(f"No video files found in {video_path}")
        print(f"📊 Found {len(videos)} video(s):")
        for v in videos:
            print(f"  - {v}")
    else:
        print(f"🎥 Processing single video: {video_path}")
        videos = [video_path]
    
    print(f"📁 Output directory: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Process each video
    all_frame_paths = []
    frame_counter = 0
    
    # Calculate frames per video
    if is_directory and len(videos) > 1:
        # For multiple videos, distribute frames
        total_target_frames = 500  # Total frames target for multiple videos
        frames_per_video = max(50, total_target_frames // len(videos))
    else:
        frames_per_video = None  # Will be calculated per video
    
    for idx, video_file in enumerate(videos):
        print(f"\n🔄 Processing video {idx+1}/{len(videos)}: {os.path.basename(video_file)}")
        
        # Get video info
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"⚠️  Warning: Cannot open video: {video_file}, skipping...")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        print(f"  Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        
        # Calculate target number of frames
        if frames_per_video:
            # Multiple videos: use calculated frames per video
            target_num_frames = min(frames_per_video, max(30, int(duration * target_fps)))
        else:
            # Single video: use standard calculation
            target_num_frames = min(1000, max(100, int(duration * target_fps)))
        
        print(f"  🎯 Target frames: {target_num_frames}")
        
        # Extract frames to a temporary directory first
        temp_dir = os.path.join(output_path, f"temp_video_{idx}")
        os.makedirs(temp_dir, exist_ok=True)
        
        print("  🔄 Extracting frames uniformly...")
        frame_paths = extract_frames_uniformly(
            video_path=video_file,
            output_dir=temp_dir,
            num_frames=target_num_frames,
            max_size=max_image_size if max_image_size > 0 else 2048
        )
        
        # Move and rename frames to combined directory with global numbering
        for frame_path in frame_paths:
            new_filename = f"{frame_counter:08d}.jpg"
            new_path = os.path.join(images_dir, new_filename)
            shutil.move(frame_path, new_path)
            all_frame_paths.append(new_path)
            frame_counter += 1
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"  ✅ Extracted {len(frame_paths)} frames")
    
    print(f"\n✅ Total frames extracted: {len(all_frame_paths)}")
    
    # Run COLMAP reconstruction
    print("🏗️  Running COLMAP reconstruction...")
    run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=colmap_config)
    
    print(f"🎉 COLMAP processing complete!")
    return output_path

# --- Add argument parsing ---
parser = argparse.ArgumentParser(
    description="Fit EDGS model to a scene from a video file or directory containing multiple videos."
)
parser.add_argument(
    "--video_path",
    type=str,
    default=os.path.join(
        project_root, "assets", "examples", "video_fruits.mp4"
    ),  # Use project_root
    help="Path to the input video file or directory containing video files.",
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
    help="Config name to use. Options: 'train' (standard), 'train_high_quality' (best quality, 12GB+ GPU), 'train_low_memory' (6-8GB GPU), 'train_very_low_memory' (4-6GB GPU). Default: train",
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
parser.add_argument(
    "--colmap_config",
    type=str,
    default="low_memory",
    choices=["high_accuracy", "balanced", "low_memory", "very_low_memory"],
    help="COLMAP configuration profile for memory vs accuracy trade-off. Options: high_accuracy (best quality, high memory), balanced (good quality, moderate memory), low_memory (reduced quality, low memory), very_low_memory (minimal quality, very low memory). Default: low_memory",
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
        print(f"Error: Video path does not exist: {args.video_path}")
        sys.exit(1)
    
    # Check if it's a directory or file
    if os.path.isdir(args.video_path):
        videos = find_videos_in_directory(args.video_path)
        if not videos:
            print(f"Error: No video files found in directory: {args.video_path}")
            sys.exit(1)
    elif not os.path.isfile(args.video_path):
        print(f"Error: Path is neither a file nor directory: {args.video_path}")
        sys.exit(1)
    
    # Set default colmap_output_path if not specified
    if not args.colmap_output_path:
        args.colmap_output_path = os.path.join(project_root, "outputs")
        
    print(f"Starting video processing for: {args.video_path}")
    try:
        # Use new uniform frame extraction that samples across entire video duration
        scene_dir = process_video_to_colmap_scene(
            video_path=args.video_path,
            output_path=args.output_path,  # Use specified output path directly
            target_fps=args.target_fps,
            max_image_size=args.max_image_size,
            colmap_config=args.colmap_config
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

# Apply max_image_size to dataset resolution if specified
if args.max_image_size > 0:
    # Convert max_image_size to resolution setting for gaussian splatting
    # The resolution parameter in GS is a scaling factor, not absolute pixels
    # We need to estimate based on typical image sizes
    if args.max_image_size <= 64:
        cfg.gs.dataset.resolution = 8  # Very aggressive downscaling
    elif args.max_image_size <= 128:
        cfg.gs.dataset.resolution = 4  # Aggressive downscaling
    elif args.max_image_size <= 256:
        cfg.gs.dataset.resolution = 3  # Moderate downscaling
    elif args.max_image_size <= 512:
        cfg.gs.dataset.resolution = 2  # Light downscaling
    elif args.max_image_size <= 1024:
        cfg.gs.dataset.resolution = 1  # Minimal downscaling
    else:
        cfg.gs.dataset.resolution = -1  # Use original resolution
    print(f"Setting resolution scale to {cfg.gs.dataset.resolution} based on max_image_size={args.max_image_size}")

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
