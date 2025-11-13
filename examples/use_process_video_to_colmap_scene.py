#!/usr/bin/env python
# coding: utf-8

"""
Example: Using process_video_to_colmap_scene as an independent library

This script demonstrates how to use the process_video_to_colmap_scene
function from source/process_video_to_colmap_scene.py in your own programs.
"""

import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from the independent module
from source.process_video_to_colmap_scene import (
    process_video_to_colmap_scene,
    check_colmap_scene,
    find_videos_in_directory,
    find_images_in_directory,
)


def example_process_single_video():
    """Example: Process a single video file to create a COLMAP scene."""
    video_path = "/path/to/video.mp4"
    output_path = "/path/to/output/colmap_scene"

    # COLMAP configuration with preprocessing settings
    colmap_cfg = {
        'preprocessing': {
            'target_fps': 3.0,          # Extract 3 frames per second
            'max_image_size': 1920,      # Resize images to max 1920px
        },
        # Add other COLMAP settings as needed
    }

    # Process video to COLMAP scene
    scene_dir = process_video_to_colmap_scene(
        video_path=video_path,
        output_path=output_path,
        colmap_cfg=colmap_cfg,
        is_360=False  # Set to True for 360° videos
    )

    print(f"COLMAP scene created at: {scene_dir}")


def example_process_360_video():
    """Example: Process a 360° video with automatic cubemap conversion."""
    video_path = "/path/to/360_video.mp4"
    output_path = "/path/to/output/colmap_scene_360"

    colmap_cfg = {
        'preprocessing': {
            'target_fps': 2.0,           # Lower FPS for 360° (5x more images)
            'max_image_size': 2048,      # Higher resolution for 360°
        },
    }

    # Process 360° video with automatic cubemap conversion
    scene_dir = process_video_to_colmap_scene(
        video_path=video_path,
        output_path=output_path,
        colmap_cfg=colmap_cfg,
        is_360=True  # Enable 360° processing
    )

    print(f"360° COLMAP scene created at: {scene_dir}")


def example_process_video_directory():
    """Example: Process all videos in a directory."""
    video_dir = "/path/to/videos"
    output_path = "/path/to/output/colmap_scene"

    colmap_cfg = {
        'preprocessing': {
            'target_fps': 2.5,
            'max_image_size': 1600,
        },
    }

    # Process all videos in directory
    scene_dir = process_video_to_colmap_scene(
        video_path=video_dir,  # Pass directory instead of single file
        output_path=output_path,
        colmap_cfg=colmap_cfg,
        is_360=False
    )

    print(f"COLMAP scene from multiple videos created at: {scene_dir}")


def example_check_and_validate():
    """Example: Check if a COLMAP scene exists before processing."""
    scene_path = "/path/to/potential/colmap/scene"

    if check_colmap_scene(scene_path):
        print(f"✅ Valid COLMAP scene found at: {scene_path}")
        print("   Skipping processing...")
    else:
        print(f"❌ No valid COLMAP scene at: {scene_path}")
        print("   Starting video processing...")
        # Process video here...


def example_find_media_files():
    """Example: Find videos and images in directories."""
    media_directory = "/path/to/media"

    # Find all videos recursively
    videos = find_videos_in_directory(media_directory)
    print(f"Found {len(videos)} video(s):")
    for video in videos[:5]:  # Show first 5
        print(f"  - {video}")

    # Find all images in a specific directory (not recursive)
    images = find_images_in_directory(media_directory)
    print(f"\nFound {len(images)} image(s) in {media_directory}")


def example_custom_config():
    """Example: Use custom COLMAP configuration."""
    video_path = "/path/to/video.mp4"
    output_path = "/path/to/output/colmap_scene"

    # Custom configuration for low-memory systems
    colmap_cfg = {
        'preprocessing': {
            'target_fps': 1.5,           # Lower FPS
            'max_image_size': 1024,      # Smaller images
        },
        'feature_extraction': {
            'max_num_features': 4096,    # Fewer features
        },
        'matching': {
            'max_num_matches': 8192,     # Fewer matches
        },
    }

    scene_dir = process_video_to_colmap_scene(
        video_path=video_path,
        output_path=output_path,
        colmap_cfg=colmap_cfg,
        is_360=False
    )

    print(f"COLMAP scene created with custom config: {scene_dir}")


if __name__ == "__main__":
    print("process_video_to_colmap_scene Usage Examples")
    print("=" * 60)
    print()
    print("This module provides functions to process videos into COLMAP")
    print("reconstruction scenes with automatic frame extraction.")
    print()
    print("Available examples:")
    print("  1. example_process_single_video() - Process one video")
    print("  2. example_process_360_video() - Process 360° video")
    print("  3. example_process_video_directory() - Process multiple videos")
    print("  4. example_check_and_validate() - Check existing scenes")
    print("  5. example_find_media_files() - Find videos/images")
    print("  6. example_custom_config() - Custom COLMAP configuration")
    print()
    print("Uncomment the example you want to run in the code.")
    print()

    # Uncomment the example you want to run:
    # example_process_single_video()
    # example_process_360_video()
    # example_process_video_directory()
    # example_check_and_validate()
    # example_find_media_files()
    # example_custom_config()
