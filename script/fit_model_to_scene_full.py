#!/usr/bin/env python
# coding: utf-8

"""
EDGS: Eliminating Densification for Gaussian Splatting

EDGS improves 3D Gaussian Splatting by removing the need for densification.
It starts from a dense point cloud initialization based on 2D correspondences, leading to:
- ⚡ Faster convergence (only 25% of training time)
- 🌀 Higher rendering quality
- 💡 No need for progressive densification

This script supports:
- Single video files (.mp4, .mov, .avi)
- Directories containing multiple videos
- Directories containing image sequences (.jpg, .png, .bmp)
"""

import argparse
import logging
import os
import random
import sys
import shutil
import subprocess
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Check and fix CUDA errors before importing torch
cuda_fix_script = os.path.join(project_root, "script", "cuda_error_fix.py")
if os.path.exists(cuda_fix_script):
    try:
        # Run CUDA fix if needed
        result = subprocess.run([sys.executable, cuda_fix_script], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Warning: CUDA initialization issue detected. Running in CPU mode.")
    except Exception as e:
        print(f"⚠️  Could not run CUDA check: {e}")

import cv2
import hydra
import numpy as np
import omegaconf
import torch
import wandb
from hydra import compose, initialize
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
from source.utils_preprocess import run_colmap_on_scene
from source.convert_equirectangular_to_cubemap import convert_equirectangular_to_cubemap
from source.process_video_to_colmap_scene import (
    process_video_to_colmap_scene,
    check_colmap_scene,
    find_videos_in_directory,
    find_images_in_directory,
    run_colmap_with_retry,
    get_min_registered_images,
)
from script.undistort_colmap_scene import undistort_colmap_scene

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def save_configs_to_output(train_config_name, colmap_config_name, model_path):
    """Save both training and colmap configuration files to output directory."""
    try:
        # Save training config
        train_config_file = f"../configs/{train_config_name}.yaml"
        train_config_path = os.path.join(os.path.dirname(__file__), train_config_file)
        if os.path.exists(train_config_path):
            dst = os.path.join(model_path, "train_config.yaml")
            shutil.copy2(train_config_path, dst)
            print(f"📋 Training config saved to: {dst}")
        else:
            print(f"⚠️ Training config file not found: {train_config_path}")
        
        # Save colmap config - handle both old and new naming
        if not colmap_config_name.startswith("colmap_"):
            colmap_config_name = f"colmap_{colmap_config_name}"
        colmap_config_file = f"../configs/{colmap_config_name}.yaml"
        colmap_config_path = os.path.join(os.path.dirname(__file__), colmap_config_file)
        if os.path.exists(colmap_config_path):
            dst = os.path.join(model_path, "colmap_config.yaml")
            shutil.copy2(colmap_config_path, dst)
            print(f"📋 COLMAP config saved to: {dst}")
        else:
            print(f"⚠️ COLMAP config file not found: {colmap_config_path}")
    except Exception as e:
        print(f"Warning: Could not copy config files: {e}")


def process_images_to_colmap_scene(image_dir, output_path, colmap_cfg):
    """
    Process a directory of images with COLMAP.

    Args:
        image_dir: Directory containing images
        output_path: Directory to save COLMAP scene
        colmap_cfg: COLMAP configuration dictionary with preprocessing settings

    Returns:
        scene_dir: Path to COLMAP scene directory
    """
    print(f"🎨 Processing images from directory: {image_dir}")

    # Find all images
    image_paths = find_images_in_directory(image_dir)
    if not image_paths:
        raise ValueError(f"No image files found in {image_dir}")

    print(f"📊 Found {len(image_paths)} images")

    # Get preprocessing settings from colmap config
    max_image_size = colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, "images")

    # Check if images already processed
    if os.path.exists(images_dir):
        existing_images = find_images_in_directory(images_dir)
        if existing_images and len(existing_images) >= len(image_paths):
            print(f"✅ Found {len(existing_images)} processed images in {images_dir}, skipping processing")

            # Calculate minimum required registered images
            min_registered = get_min_registered_images(images_dir)

            # Check if COLMAP has already been run with sufficient quality
            if check_colmap_scene(output_path, min_registered_images=min_registered):
                print(f"✅ COLMAP reconstruction already exists with sufficient registered images")
                return output_path
            else:
                print("🏗️  Running COLMAP reconstruction on existing images...")
                if run_colmap_with_retry(output_path, colmap_cfg, images_dir):
                    print(f"🎉 COLMAP processing complete!")
                    return output_path
                else:
                    raise RuntimeError("COLMAP reconstruction failed - see suggestions above")

    os.makedirs(images_dir, exist_ok=True)
    
    # Copy and optionally resize images
    print("🔄 Processing images...")
    for idx, src_path in enumerate(image_paths):
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Could not read image {src_path}")
            continue
        
        # Resize if needed
        if max_image_size > 0:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > max_image_size:
                scale = max_image_size / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save to output directory with sequential naming
        dst_filename = f"{idx:08d}.jpg"
        dst_path = os.path.join(images_dir, dst_filename)
        cv2.imwrite(dst_path, img)
    
    print(f"✅ Processed {len(image_paths)} images")

    # Run COLMAP reconstruction with retry logic
    print("🏗️  Running COLMAP reconstruction...")
    if run_colmap_with_retry(output_path, colmap_cfg, images_dir):
        print(f"🎉 COLMAP processing complete!")
        return output_path
    else:
        raise RuntimeError("COLMAP reconstruction failed - see suggestions above")


def check_existing_colmap_scene(colmap_path):
    """
    Check if a path contains a valid COLMAP scene.
    
    Args:
        colmap_path: Path to check
        
    Returns:
        bool: True if valid COLMAP scene exists
    """
    if not colmap_path or not os.path.exists(colmap_path):
        return False
    
    sparse_dir = os.path.join(colmap_path, "sparse", "0")
    if not os.path.exists(sparse_dir):
        return False
    
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(sparse_dir, f))]
    
    return len(missing_files) == 0


def setup_cuda():
    """Initialize and check CUDA availability."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This model requires GPU.")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    
    # Initialize CUDA to avoid lazy initialization issues
    torch.cuda.init()
    torch.cuda.empty_cache()


def initialize_wandb(cfg):
    """Initialize Weights & Biases logging if enabled."""
    if cfg.wandb.mode != "disabled":
        logging.info(f"WandB logging enabled (mode={cfg.wandb.mode})")
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
        )
    else:
        logging.info("WandB logging disabled")


def visualize_initial_views(trainer, model_path, num_views=4):
    """Visualize initial rendered views before training."""
    with torch.no_grad():
        available_cams = trainer.GS.scene.getTrainCameras()
        num_cams_to_viz = min(num_views, len(available_cams))
        viewpoint_cams = random.sample(available_cams, num_cams_to_viz)
        
        for idx, viewpoint_cam in enumerate(viewpoint_cams):
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            
            image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
            image_gt_np = viewpoint_cam.original_image.clone().detach().cpu().numpy().transpose(1, 2, 0)
            
            # Clip values to [0, 255]
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            ax[0].imshow(image_gt_np)
            ax[0].set_title("Ground Truth")
            ax[0].axis("off")
            ax[1].imshow(image_np)
            ax[1].set_title("Initial Render")
            ax[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f"viewpoint_{idx}_initial.png"))
            plt.close(fig)
        
        return viewpoint_cams


def visualize_final_views(trainer, viewpoint_cams):
    """Visualize final rendered views after training."""
    with torch.no_grad():
        for viewpoint_cam in viewpoint_cams:
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            
            image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
            image_gt_np = viewpoint_cam.original_image.clone().detach().cpu().numpy().transpose(1, 2, 0)
            
            # Clip values to [0, 255]
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)
            
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            ax[0].imshow(image_gt_np)
            ax[0].set_title("Ground Truth")
            ax[0].axis("off")
            ax[1].imshow(image_np)
            ax[1].set_title("Final Render")
            ax[1].axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit EDGS model to a scene from video/images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(project_root, "assets", "examples", "video_fruits.mp4"),
        help="Path to input: COLMAP scene, image directory, video directory, or single video file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="Config name (e.g., train, train_01_highest_quality, train_low_memory)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save EDGS training results.",
    )
    parser.add_argument(
        "--colmap_config",
        type=str,
        default="colmap_03_optimal_quality",
        choices=["colmap_01_highest_quality", "colmap_02_high_quality", "colmap_03_optimal_quality",
                 "colmap_04_medium_quality", "colmap_05_low_quality", "colmap_06_lowest_quality",
                 "colmap_07_360_optimized", "colmap_08_360_2fps"],
        help="COLMAP configuration profile (01=highest to 06=lowest quality, 07/08=360 optimized).",
    )
    parser.add_argument(
        "--360",
        action="store_true",
        help="Process 360 degree equirectangular video by converting to cubemap faces.",
    )
    parser.add_argument(
        "--fov-360",
        type=float,
        default=120,
        help="Field of view for 360 perspective conversion (default 120°, ~25%% overlap).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of video frames to extract. For 360: total images = max_frames × 5.",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help="Hydra config overrides (e.g., train.gs_epochs=60000 train.no_densify=True).",
    )
    return parser.parse_args()


def load_colmap_config(config_name):
    """Load COLMAP configuration from file."""
    # Handle both old and new naming conventions
    if not config_name.startswith("colmap_"):
        config_name = f"colmap_{config_name}"
    
    config_path = os.path.join(project_root, "configs", f"{config_name}.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name  # Store the config name
    return config

def prepare_scene_directory(args, colmap_cfg):
    """
    Prepare the COLMAP scene directory, either using existing or creating new.
    
    Args:
        args: Command line arguments
        colmap_cfg: COLMAP configuration dictionary
        
    Returns:
        scene_dir: Path to COLMAP scene directory
    """
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Set default base output path for generated COLMAP scenes
    base_output_path = args.output_path or os.path.join(project_root, "outputs")
    
    # Process based on input type
    if os.path.isdir(args.input):
        # Priority 1: Check if it's a COLMAP scene
        if check_colmap_scene(args.input):
            print(f"✅ Found existing COLMAP scene: {args.input}")
            # Check if undistortion is needed and perform it
            scene_dir = undistort_colmap_scene(args.input)
            return scene_dir
        
        # Priority 2: Check for images in the directory
        images = find_images_in_directory(args.input)
        if images:
            print(f"📸 Found {len(images)} images in directory")
            scene_dir = process_images_to_colmap_scene(
                image_dir=args.input,
                output_path=os.path.join(base_output_path, "image_scene"),
                colmap_cfg=colmap_cfg
            )
            return scene_dir
        
        # Priority 3: Check for videos in the directory
        videos = find_videos_in_directory(args.input)
        if videos:
            print(f"🎥 Found {len(videos)} video(s) in directory")
            # Use args directly with attribute name matching the CLI flag
            is_360_mode = getattr(args, '360', False)
            if is_360_mode:
                print("🌐 360 degree video mode enabled")
            fov_360 = getattr(args, 'fov_360', 120)
            max_frames = getattr(args, 'max_frames', None)
            scene_dir = process_video_to_colmap_scene(
                video_path=args.input,
                output_path=os.path.join(base_output_path, "video_scene"),
                colmap_cfg=colmap_cfg,
                is_360=is_360_mode,
                fov_360=fov_360,
                max_frames=max_frames
            )
            return scene_dir

        print(f"ERROR: No COLMAP scene, images, or videos found in: {args.input}")
        sys.exit(1)

    else:
        # Single file - assume it's a video
        if not args.input.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
            print(f"⚠️ Warning: {args.input} may not be a video file, attempting to process anyway...")

        # Use args directly with attribute name matching the CLI flag
        is_360_mode = getattr(args, '360', False)
        if is_360_mode:
            print("🌐 360 degree video mode enabled")
        fov_360 = getattr(args, 'fov_360', 120)
        max_frames = getattr(args, 'max_frames', None)
        scene_dir = process_video_to_colmap_scene(
            video_path=args.input,
            output_path=os.path.join(base_output_path, "video_scene"),
            colmap_cfg=colmap_cfg,
            is_360=is_360_mode,
            fov_360=fov_360,
            max_frames=max_frames
        )
        return scene_dir


def check_edgs_training_complete(model_path):
    """
    Check if EDGS training has already completed.

    Args:
        model_path: Path to EDGS output directory

    Returns:
        bool: True if training is complete
    """
    if not os.path.exists(model_path):
        return False

    # Check for common model output files
    point_cloud_file = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    checkpoint_file = os.path.join(model_path, "chkpnt30000.pth")

    # Look for any iteration checkpoint
    if os.path.exists(os.path.join(model_path, "point_cloud")):
        point_cloud_dirs = [d for d in os.listdir(os.path.join(model_path, "point_cloud"))
                           if d.startswith("iteration_")]
        if point_cloud_dirs:
            # Check if final iteration exists
            iterations = [int(d.replace("iteration_", "")) for d in point_cloud_dirs]
            max_iter = max(iterations)
            if max_iter >= 30000:  # Typical final iteration
                return True

    return os.path.exists(point_cloud_file) or os.path.exists(checkpoint_file)


def configure_resolution(cfg, colmap_cfg):
    """Configure the resolution parameter based on max_image_size from colmap config."""
    max_image_size = colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)
    if max_image_size <= 0:
        return

    if max_image_size <= 64:
        cfg.gs.dataset.resolution = 8
    elif max_image_size <= 128:
        cfg.gs.dataset.resolution = 4
    elif max_image_size <= 256:
        cfg.gs.dataset.resolution = 3
    elif max_image_size <= 512:
        cfg.gs.dataset.resolution = 2
    elif max_image_size <= 1024:
        cfg.gs.dataset.resolution = 1
    else:
        cfg.gs.dataset.resolution = -1

    print(f"📐 Resolution scale: {cfg.gs.dataset.resolution} (for max_image_size={max_image_size})")


def main():
    """Main function to orchestrate the EDGS training pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Auto-select 360-optimized config when --360 mode is enabled
    is_360_mode = getattr(args, '360', False)
    colmap_config_name = args.colmap_config
    if is_360_mode and colmap_config_name == "colmap_03_optimal_quality":
        # Use 360-optimized config as default for 360 mode
        colmap_config_name = "colmap_07_360_optimized"
        print("🌐 360 mode detected - using colmap_07_360_optimized config")

    # Load COLMAP configuration
    colmap_cfg = load_colmap_config(colmap_config_name)
    print(f"\n📋 Using COLMAP config: {colmap_config_name}")
    print(f"  - Target FPS: {colmap_cfg.get('preprocessing', {}).get('target_fps', 3.0)}")
    print(f"  - Max image size: {colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)}")
    if is_360_mode:
        print(f"  - Single camera mode: {colmap_cfg.get('processing_360', {}).get('single_camera', False)}")
    
    # Initialize Hydra configuration
    with initialize(config_path="../configs", version_base="1.1"):
        cfg = compose(config_name=args.config, overrides=args.overrides)
        print(f"\n📋 Using training config: {args.config}")
        if args.overrides:
            print(f"  Overrides: {args.overrides}")
        
        # Check if using memory-optimized config
        if any(x in args.config for x in ["_04_", "_05_", "_06_", "medium", "low"]):
            print("💾 Memory-optimized mode enabled")
    
    # Prepare COLMAP scene
    scene_dir = prepare_scene_directory(args, colmap_cfg)
    
    # Set up paths for EDGS
    cfg.gs.dataset.source_path = scene_dir
    
    if args.output_path:
        model_path = args.output_path
    else:
        model_path = os.path.join(scene_dir, "models")
    
    cfg.gs.dataset.model_path = model_path
    print(f"\n📁 COLMAP scene: {scene_dir}")
    print(f"📦 EDGS output: {model_path}")
    os.makedirs(model_path, exist_ok=True)

    # Check if EDGS training has already completed
    if check_edgs_training_complete(model_path):
        print("\n✅ EDGS training already completed!")
        print(f"Model exists at: {model_path}")
        print("\nTo retrain, delete the model directory and run again.")
        return

    # Save both config files for reference
    save_configs_to_output(args.config, args.colmap_config, model_path)

    # Initialize WandB
    initialize_wandb(cfg)
    omegaconf.OmegaConf.resolve(cfg)

    # Configure resolution
    configure_resolution(cfg, colmap_cfg)

    # Set random seed
    set_seed(cfg.seed)

    # Setup CUDA
    setup_cuda()

    # Initialize Gaussian Splatting model
    try:
        gs = hydra.utils.instantiate(cfg.gs)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("\n❌ CUDA initialization error!")
            print("Please try:")
            print("1. Restart Docker: docker compose restart")
            print("2. Check GPU: nvidia-smi")
            print(f"\nError: {e}")
            sys.exit(1)
        raise

    # Check if COLMAP reconstruction has enough cameras
    train_cameras = gs.scene.getTrainCameras()
    num_cameras = len(train_cameras)
    print(f"\n📷 COLMAP reconstruction contains {num_cameras} camera(s)")

    if num_cameras < 2:
        print("\n❌ ERROR: COLMAP reconstruction failed or produced insufficient results")
        print(f"   Found only {num_cameras} camera(s), but EDGS requires at least 2\n")
        print("Possible causes:")
        print("  1. Not enough frames extracted from video")
        print("  2. Insufficient feature matches between frames")
        print("  3. Scene has too much motion blur or lacks texture")
        print("  4. Video is too short or has little camera movement\n")
        print("Suggestions:")
        images_dir = os.path.join(scene_dir, "images")
        if os.path.exists(images_dir):
            num_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  - Images extracted: {num_images}")
            if num_images < 10:
                print(f"    → Try higher target_fps in colmap_config (current: {colmap_cfg.get('preprocessing', {}).get('target_fps', 'unknown')})")
        print(f"  - Try a different COLMAP config (current: {args.colmap_config})")
        print(f"  - Ensure video has enough camera movement and overlap between frames")
        print(f"  - Check COLMAP logs in: {scene_dir}")
        sys.exit(1)

    # Initialize trainer
    trainer = EDGSTrainer(
        GS=gs,
        training_config=cfg.gs.opt,
        device=cfg.device,
        log_wandb=(cfg.wandb.mode != "disabled"),
    )

    # Initialize with correspondence matching
    print("\n🔧 Initializing with correspondence matching...")
    trainer.timer.start()
    try:
        trainer.init_with_corr(cfg.init_wC)
    except RuntimeError as e:
        if "COLMAP reconstruction produced only" in str(e):
            print(f"\n❌ {e}")
            sys.exit(1)
        raise
    trainer.timer.pause()

    # Visualize initial views
    print("📸 Generating initial visualizations...")
    viewpoint_cams = visualize_initial_views(trainer, model_path)

    # Train the model
    print("\n🚀 Starting EDGS optimization...")
    # Keep default save_iterations from config for intermediate checkpoints
    trainer.train(cfg.train)

    # Visualize final views
    print("📸 Generating final visualizations...")
    visualize_final_views(trainer, viewpoint_cams)

    # Save the final model
    print("\n💾 Saving final model...")
    trainer.save_model()

    print("\n✨ Training complete!")
    print(f"Results saved to: {model_path}")


if __name__ == "__main__":
    main()