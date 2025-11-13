#!/usr/bin/env python
# coding: utf-8

"""
Process videos to COLMAP scenes with frame extraction.

This module provides functionality to process video files (including 360° videos)
into COLMAP reconstruction scenes with uniform frame extraction.
"""

import os
import shutil
from pathlib import Path

import cv2

from source.utils_preprocess import run_colmap_on_scene
from source.convert_equirectangular_to_cubemap import convert_equirectangular_to_cubemap


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
    # Import here to avoid circular dependency
    from script.extract_frames_uniform import extract_frames_uniformly, get_video_duration_safe

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
