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
from pathlib import Path

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
from script.extract_frames_uniform import extract_frames_uniformly, get_video_duration_safe
from script.undistort_colmap_scene import needs_undistortion, undistort_colmap_scene

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def check_colmap_scene(directory_path):
    """
    Check if a directory contains a valid COLMAP scene.
    
    Args:
        directory_path: Directory to check
        
    Returns:
        bool: True if valid COLMAP scene exists
    """
    if not os.path.isdir(directory_path):
        return False
    
    # Check for required COLMAP directories/files
    sparse_dir = os.path.join(directory_path, "sparse")
    if not os.path.exists(sparse_dir):
        return False
    
    # Check for model folders (0, 1, etc.) or direct files
    has_model = False
    
    # Check numbered model directories
    for i in range(10):  # Check first 10 possible model folders
        model_dir = os.path.join(sparse_dir, str(i))
        if os.path.exists(model_dir):
            # Check for essential COLMAP files
            cameras = os.path.join(model_dir, "cameras.bin")
            images = os.path.join(model_dir, "images.bin")
            points = os.path.join(model_dir, "points3D.bin")
            
            cameras_txt = os.path.join(model_dir, "cameras.txt")
            images_txt = os.path.join(model_dir, "images.txt")
            points_txt = os.path.join(model_dir, "points3D.txt")
            
            if (os.path.exists(cameras) and os.path.exists(images)) or \
               (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
                has_model = True
                break
    
    # Also check for direct files in sparse directory
    if not has_model:
        cameras = os.path.join(sparse_dir, "cameras.bin")
        images = os.path.join(sparse_dir, "images.bin")
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")
        
        if (os.path.exists(cameras) and os.path.exists(images)) or \
           (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
            has_model = True
    
    return has_model


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


def find_images_in_directory(directory_path, extensions=('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP')):
    """
    Find all image files in a directory.
    
    Args:
        directory_path: Directory to search
        extensions: Tuple of valid image file extensions
        
    Returns:
        List of image paths
    """
    images = []
    directory_path = Path(directory_path)
    
    # Only search in the immediate directory, not subdirectories
    for file_path in directory_path.glob('*'):
        if file_path.suffix in extensions:
            images.append(str(file_path))
    
    return sorted(images)


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

            # Check if COLMAP has already been run
            if check_colmap_scene(output_path):
                print(f"✅ COLMAP reconstruction already exists, skipping COLMAP stage")
                return output_path
            else:
                print("🏗️  Running COLMAP reconstruction on existing images...")
                run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=colmap_cfg)
                print(f"🎉 COLMAP processing complete!")
                return output_path

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
    
    # Run COLMAP reconstruction
    print("🏗️  Running COLMAP reconstruction...")
    run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=colmap_cfg)
    
    print(f"🎉 COLMAP processing complete!")
    return output_path


def convert_equirectangular_to_cubemap(input_image_path, output_dir):
    """
    Convert an equirectangular 360 image to 5 cubemap face images.

    Note: Bottom face is excluded to avoid capturing the camera operator.

    Args:
        input_image_path: Path to equirectangular image
        output_dir: Directory to save cubemap face images

    Returns:
        List of paths to the 5 generated cubemap face images (front, right, back, left, top)
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Define the 5 cubemap faces with their respective ffmpeg parameters
    # Format: (suffix, yaw, pitch, roll)
    # Note: Bottom face is excluded as it often captures the camera operator
    faces = [
        ("front", 0, 0, 0),      # Front face
        ("right", -90, 0, 0),    # Right face
        ("back", 180, 0, 0),     # Back face
        ("left", 90, 0, 0),      # Left face
        ("top", 0, 90, 0),       # Top face (pitch=90 looks up)
    ]

    output_paths = []

    for face_name, yaw, pitch, roll in faces:
        output_path = os.path.join(output_dir, f"{basename}_{face_name}.png")

        # Build ffmpeg command for v360 filter
        # e:rectilinear converts equirectangular to rectilinear projection
        # h_fov and v_fov set the field of view to 90 degrees for cube face
        cmd = [
            "ffmpeg",
            "-i", input_image_path,
            "-vf", f"v360=e:rectilinear:h_fov=90:v_fov=90:yaw={yaw}:pitch={pitch}:roll={roll}",
            "-y",  # Overwrite output files
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            output_paths.append(output_path)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Warning: Failed to generate {face_name} face: {e.stderr}")
            continue

    return output_paths


def process_video_to_colmap_scene(video_path, output_path, colmap_cfg, is_360=False):
    """
    Process video(s) with uniform frame extraction and run COLMAP.

    Args:
        video_path: Path to video file or directory
        output_path: Directory to save COLMAP scene
        colmap_cfg: COLMAP configuration dictionary with preprocessing settings
        is_360: If True, process as 360 degree equirectangular video

    Returns:
        scene_dir: Path to COLMAP scene directory
    """
    # Check if input is a directory or single video file
    is_directory = os.path.isdir(video_path)

    if is_directory:
        print(f"🎥 Processing videos from directory: {video_path}")
        videos = find_videos_in_directory(video_path)
        if not videos:
            raise ValueError(f"No video files found in {video_path}")
        print(f"📊 Found {len(videos)} video(s)")
    else:
        print(f"🎥 Processing single video: {video_path}")
        videos = [video_path]

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, "images")

    # Check if images already exist
    if os.path.exists(images_dir):
        existing_images = find_images_in_directory(images_dir)
        if existing_images:
            print(f"✅ Found {len(existing_images)} existing images in {images_dir}, skipping extraction")

            # Check if COLMAP has already been run
            if check_colmap_scene(output_path):
                print(f"✅ COLMAP reconstruction already exists, skipping COLMAP stage")
                return output_path
            else:
                print("🏗️  Running COLMAP reconstruction on existing images...")
                run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=colmap_cfg)
                print(f"🎉 COLMAP processing complete!")
                return output_path

    os.makedirs(images_dir, exist_ok=True)
    
    # Process each video
    all_frame_paths = []
    frame_counter = 0
    
    # Calculate frames per video for multiple videos
    if is_directory and len(videos) > 1:
        total_target_frames = 500  # Total frames target for multiple videos
        frames_per_video = max(50, total_target_frames // len(videos))
    else:
        frames_per_video = None
    
    for idx, video_file in enumerate(videos):
        print(f"\n🔄 Processing video {idx+1}/{len(videos)}: {os.path.basename(video_file)}")
        
        # Let extract_frames_uniformly handle all the video metadata checking
        # It has robust handling for corrupted metadata
        if frames_per_video:
            target_num_frames = frames_per_video
        else:
            # Get actual video duration and calculate frames based on target_fps
            duration = get_video_duration_safe(video_file)
            target_fps = colmap_cfg.get('preprocessing', {}).get('target_fps', 3.0)
            
            if duration and duration > 0:
                # Calculate based on ACTUAL duration, not 3 minutes!
                target_num_frames = int(target_fps * duration)
                print(f"  📊 Video duration: {duration:.1f}s, extracting {target_num_frames} frames at {target_fps} fps")
            else:
                # If we can't get duration, use a reasonable default
                target_num_frames = 300
                print(f"  ⚠️ Could not determine video duration, using default {target_num_frames} frames")
        
        print(f"  🎯 Requesting {target_num_frames} frames")
        
        # Extract frames to a temporary directory
        temp_dir = os.path.join(output_path, f"temp_video_{idx}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            print("  🔄 Extracting frames uniformly...")
            max_image_size = colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)
            frame_paths = extract_frames_uniformly(
                video_path=video_file,
                output_dir=temp_dir,
                num_frames=target_num_frames,
                max_size=max_image_size if max_image_size > 0 else 2048
            )
            
            # Process frames: convert to cubemap if 360 mode, otherwise just move
            if is_360:
                print(f"  🌐 Converting {len(frame_paths)} equirectangular frames to cubemap faces...")
                cubemap_temp_dir = os.path.join(output_path, f"temp_cubemap_{idx}")
                os.makedirs(cubemap_temp_dir, exist_ok=True)

                for frame_path in frame_paths:
                    # Convert each equirectangular frame to 6 cubemap faces
                    cubemap_faces = convert_equirectangular_to_cubemap(frame_path, cubemap_temp_dir)

                    # Move cubemap faces to combined directory with global numbering
                    for face_path in cubemap_faces:
                        new_filename = f"{frame_counter:08d}.png"
                        new_path = os.path.join(images_dir, new_filename)
                        shutil.move(face_path, new_path)
                        all_frame_paths.append(new_path)
                        frame_counter += 1

                # Clean up cubemap temp directory
                if os.path.exists(cubemap_temp_dir):
                    shutil.rmtree(cubemap_temp_dir)

                print(f"  ✅ Converted {len(frame_paths)} frames to {len(frame_paths) * 5} cubemap faces (excluding bottom)")
            else:
                # Move frames to combined directory with global numbering
                for frame_path in frame_paths:
                    new_filename = f"{frame_counter:08d}.png"
                    new_path = os.path.join(images_dir, new_filename)
                    shutil.move(frame_path, new_path)
                    all_frame_paths.append(new_path)
                    frame_counter += 1

                print(f"  ✅ Extracted {len(frame_paths)} frames")
            
        except Exception as e:
            print(f"  ❌ Failed to extract frames: {e}")
            continue
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    if not all_frame_paths:
        raise RuntimeError("No frames were extracted from any video")
    
    print(f"\n✅ Total frames extracted: {len(all_frame_paths)}")
    
    # Run COLMAP reconstruction
    print("🏗️  Running COLMAP reconstruction...")
    run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=colmap_cfg)
    
    print(f"🎉 COLMAP processing complete!")
    return output_path


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
                 "colmap_04_medium_quality", "colmap_05_low_quality", "colmap_06_lowest_quality"],
        help="COLMAP configuration profile (01=highest to 06=lowest quality).",
    )
    parser.add_argument(
        "--360",
        action="store_true",
        help="Process 360 degree equirectangular video by converting to cubemap faces.",
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
            scene_dir = process_video_to_colmap_scene(
                video_path=args.input,
                output_path=os.path.join(base_output_path, "video_scene"),
                colmap_cfg=colmap_cfg,
                is_360=is_360_mode
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
        scene_dir = process_video_to_colmap_scene(
            video_path=args.input,
            output_path=os.path.join(base_output_path, "video_scene"),
            colmap_cfg=colmap_cfg,
            is_360=is_360_mode
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
    
    # Load COLMAP configuration
    colmap_cfg = load_colmap_config(args.colmap_config)
    print(f"\n📋 Using COLMAP config: {args.colmap_config}")
    print(f"  - Target FPS: {colmap_cfg.get('preprocessing', {}).get('target_fps', 3.0)}")
    print(f"  - Max image size: {colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)}")
    
    # Initialize Hydra configuration
    with initialize(config_path="../configs", version_base="1.1"):
        cfg = compose(config_name=args.config)
        print(f"\n📋 Using training config: {args.config}")
        
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
    trainer.init_with_corr(cfg.init_wC)
    trainer.timer.pause()

    # Visualize initial views
    print("📸 Generating initial visualizations...")
    viewpoint_cams = visualize_initial_views(trainer, model_path)

    # Train the model
    print("\n🚀 Starting EDGS optimization...")
    trainer.saving_iterations = []
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