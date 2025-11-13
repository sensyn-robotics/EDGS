#!/usr/bin/env python
# coding: utf-8

"""
Convert equirectangular 360° images to cubemap faces.

This module provides functionality to convert 360 degree equirectangular images
into cubemap face projections using ffmpeg's v360 filter.
"""

import os
import subprocess


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
