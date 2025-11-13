#!/usr/bin/env python
# coding: utf-8

"""
Example: Using convert_equirectangular_to_cubemap as an independent library

This script demonstrates how to use the convert_equirectangular_to_cubemap
function from source/convert_equirectangular_to_cubemap.py in your own programs.
"""

import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the function from the independent module
from source.convert_equirectangular_to_cubemap import convert_equirectangular_to_cubemap


def example_basic_usage():
    """Basic example: Convert a single 360° image to cubemap faces."""
    input_image = "/path/to/360_image.jpg"
    output_dir = "/path/to/output/cubemap_faces"

    # Convert equirectangular image to 5 cubemap faces (excluding bottom)
    cubemap_faces = convert_equirectangular_to_cubemap(input_image, output_dir)

    print(f"Generated {len(cubemap_faces)} cubemap faces:")
    for face in cubemap_faces:
        print(f"  - {face}")


def example_batch_processing():
    """Example: Process multiple 360° images."""
    import glob

    input_dir = "/path/to/360_images"
    output_base_dir = "/path/to/output"

    # Find all 360° images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))

    for image_file in image_files:
        # Create a subdirectory for each image's cubemap faces
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        output_dir = os.path.join(output_base_dir, image_name)

        print(f"Processing {image_name}...")
        cubemap_faces = convert_equirectangular_to_cubemap(image_file, output_dir)
        print(f"  Generated {len(cubemap_faces)} faces")


def example_custom_processing():
    """Example: Convert and then do custom processing on cubemap faces."""
    input_image = "/path/to/360_image.jpg"
    output_dir = "/path/to/output/cubemap_faces"

    # Convert to cubemap
    cubemap_faces = convert_equirectangular_to_cubemap(input_image, output_dir)

    # Do something with each face
    for face_path in cubemap_faces:
        face_name = os.path.basename(face_path)
        print(f"Processing {face_name}...")

        # Example: Load and process the image
        # import cv2
        # img = cv2.imread(face_path)
        # # Do custom processing...
        # cv2.imwrite(face_path, img)


if __name__ == "__main__":
    print("convert_equirectangular_to_cubemap Usage Examples")
    print("=" * 60)
    print()
    print("This module provides a function to convert 360° equirectangular")
    print("images into cubemap face projections.")
    print()
    print("Available examples:")
    print("  1. example_basic_usage() - Convert a single image")
    print("  2. example_batch_processing() - Process multiple images")
    print("  3. example_custom_processing() - Convert and process faces")
    print()
    print("Uncomment the example you want to run in the code.")
    print()

    # Uncomment the example you want to run:
    # example_basic_usage()
    # example_batch_processing()
    # example_custom_processing()
