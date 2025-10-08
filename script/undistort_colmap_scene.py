#!/usr/bin/env python3
"""
Undistort a COLMAP reconstruction to convert distorted camera models
(OPENCV, RADIAL, SIMPLE_RADIAL, etc.) to PINHOLE models with undistorted images.

This script uses pycolmap's undistort_images function to properly undistort
both the images and camera parameters.
"""

import argparse
import os
import shutil
import sys

import pycolmap


def needs_undistortion(scene_path):
    """
    Check if a COLMAP scene needs undistortion.

    Args:
        scene_path: Path to COLMAP scene directory (contains sparse/0)

    Returns:
        bool: True if any camera uses a distorted model
    """
    reconstruction_path = os.path.join(scene_path, "sparse", "0")
    if not os.path.exists(reconstruction_path):
        return False

    try:
        reconstruction = pycolmap.Reconstruction(reconstruction_path)

        for cam in reconstruction.cameras.values():
            # Check if camera model is distorted
            if cam.model.name not in ["PINHOLE", "SIMPLE_PINHOLE"]:
                return True

        return False
    except Exception as e:
        print(f"Error reading reconstruction: {e}")
        return False


def undistort_colmap_scene(input_scene, output_scene=None, force=False):
    """
    Undistort a COLMAP scene with distorted cameras.

    Args:
        input_scene: Path to COLMAP scene directory (contains sparse/0 and images/)
        output_scene: Path to output directory (default: input_scene + '_undistorted')
        force: Force undistortion even if cameras are already PINHOLE

    Returns:
        str: Path to undistorted scene (or original if no undistortion needed)
    """
    # Set up paths
    input_sparse = os.path.join(input_scene, "sparse", "0")
    input_images = os.path.join(input_scene, "images")

    # Check if input exists
    if not os.path.exists(input_sparse):
        print(f"❌ Error: Sparse reconstruction not found: {input_sparse}")
        sys.exit(1)

    if not os.path.exists(input_images):
        print(f"❌ Error: Images directory not found: {input_images}")
        sys.exit(1)

    # Check if undistortion is needed
    if not force and not needs_undistortion(input_scene):
        print("✅ All cameras are already PINHOLE/SIMPLE_PINHOLE - no undistortion needed!")
        return input_scene

    # Set output path
    if output_scene is None:
        output_scene = input_scene.rstrip('/') + '_undistorted'

    print(f"📂 Input scene: {input_scene}")
    print(f"📂 Output scene: {output_scene}")
    print(f"🔧 Undistorting images and converting cameras to PINHOLE...")

    # Create output directory
    os.makedirs(output_scene, exist_ok=True)

    try:
        # Use pycolmap's undistort_images function
        pycolmap.undistort_images(
            output_path=output_scene,
            input_path=input_sparse,
            image_path=input_images,
            output_type="COLMAP"
        )

        print(f"\n✅ Undistortion complete!")
        print(f"📂 Undistorted scene: {output_scene}")

        # Move files from sparse/ to sparse/0/ for compatibility
        sparse_dir = os.path.join(output_scene, "sparse")
        sparse_0_dir = os.path.join(output_scene, "sparse", "0")

        # Check if files are directly in sparse/ (pycolmap output format)
        if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
            os.makedirs(sparse_0_dir, exist_ok=True)
            for file in ["cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"]:
                src = os.path.join(sparse_dir, file)
                dst = os.path.join(sparse_0_dir, file)
                if os.path.exists(src):
                    shutil.move(src, dst)
            print(f"📁 Moved reconstruction files to sparse/0/")

        # Verify output
        output_sparse = os.path.join(output_scene, "sparse", "0")
        if os.path.exists(output_sparse):
            reconstruction = pycolmap.Reconstruction(output_sparse)
            print(f"📊 Output: {len(reconstruction.cameras)} cameras, "
                  f"{len(reconstruction.images)} images, "
                  f"{len(reconstruction.points3D)} points")

            # Verify all cameras are now PINHOLE
            all_pinhole = all(cam.model.name in ["PINHOLE", "SIMPLE_PINHOLE"]
                            for cam in reconstruction.cameras.values())
            if all_pinhole:
                print("✅ All cameras successfully converted to PINHOLE")
            else:
                print("⚠️  Warning: Some cameras are still not PINHOLE")

        return output_scene

    except Exception as e:
        print(f"❌ Error during undistortion: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Undistort COLMAP scene to convert all cameras to PINHOLE model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Undistort a scene (output to scene_undistorted/)
  python undistort_colmap_scene.py data/my_scene

  # Specify custom output path
  python undistort_colmap_scene.py data/my_scene --output data/my_scene_clean

  # Force undistortion even if already PINHOLE
  python undistort_colmap_scene.py data/my_scene --force
        """
    )
    parser.add_argument(
        "input_scene",
        type=str,
        help="Path to input COLMAP scene directory (contains sparse/0 and images/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output scene directory (default: input_scene + '_undistorted')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force undistortion even if cameras are already PINHOLE"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_scene):
        print(f"❌ Error: Input scene does not exist: {args.input_scene}")
        sys.exit(1)

    output_scene = undistort_colmap_scene(args.input_scene, args.output, args.force)
    print(f"\n💡 Use this path for training: {output_scene}")


if __name__ == "__main__":
    main()
