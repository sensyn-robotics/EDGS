#!/usr/bin/env python
# coding: utf-8

"""
Example: Using video processing utilities as a library

This script demonstrates how to use the video processing functions
from source.utils_video as a library in your own scripts.
"""

import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.utils_video import (
    convert_equirectangular_to_cubemap,
    process_video_to_colmap_scene,
    check_colmap_scene,
    find_videos_in_directory,
    find_images_in_directory,
)


def example_convert_360_image():
    """Example: Convert a 360° equirectangular image to cubemap faces."""
    input_image = "/path/to/360_image.jpg"
    output_dir = "/path/to/output/cubemap_faces"

    # Convert equirectangular image to 5 cubemap faces (excluding bottom)
    cubemap_faces = convert_equirectangular_to_cubemap(input_image, output_dir)

    print(f"Generated {len(cubemap_faces)} cubemap faces:")
    for face in cubemap_faces:
        print(f"  - {face}")


def example_process_video():
    """Example: Process a video file to create a COLMAP scene."""
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
    """Example: Process a 360° video to COLMAP scene with cubemap conversion."""
    video_path = "/path/to/360_video.mp4"
    output_path = "/path/to/output/colmap_scene_360"

    colmap_cfg = {
        'preprocessing': {
            'target_fps': 2.0,
            'max_image_size': 2048,
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


def example_check_colmap_scene():
    """Example: Check if a directory contains a valid COLMAP scene."""
    scene_path = "/path/to/potential/colmap/scene"

    if check_colmap_scene(scene_path):
        print(f"✅ Valid COLMAP scene found at: {scene_path}")
    else:
        print(f"❌ No valid COLMAP scene at: {scene_path}")


def example_find_media_files():
    """Example: Find videos and images in a directory."""
    directory = "/path/to/media/directory"

    # Find all videos recursively
    videos = find_videos_in_directory(directory)
    print(f"Found {len(videos)} video(s):")
    for video in videos:
        print(f"  - {video}")

    # Find all images in the directory (not recursive)
    images = find_images_in_directory(directory)
    print(f"\nFound {len(images)} image(s):")
    for image in images[:5]:  # Show first 5
        print(f"  - {image}")


if __name__ == "__main__":
    print("Video Processing Utilities Examples")
    print("=" * 50)
    print()
    print("This script shows examples of using source.utils_video functions.")
    print("Uncomment the example functions below to run them.")
    print()

    # Uncomment the example you want to run:
    # example_convert_360_image()
    # example_process_video()
    # example_process_360_video()
    # example_check_colmap_scene()
    # example_find_media_files()
