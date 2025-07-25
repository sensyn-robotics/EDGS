#!/usr/bin/env python3
"""
Undistort COLMAP dataset using pycolmap for EDGS training.

This script converts COLMAP datasets with distorted camera models (like SIMPLE_RADIAL)
to undistorted PINHOLE camera models that are compatible with Gaussian Splatting.

Usage:
    python script/undistort_colmap.py <input_path> <output_path>

Example:
    python script/undistort_colmap.py data/colmap_otowa data/colmap_otowa_undistorted
"""

import sys
import pycolmap
from pathlib import Path


def undistort_colmap_dataset(input_path: str, output_path: str):
    """
    Undistort a COLMAP dataset using pycolmap.
    
    Args:
        input_path: Path to the input COLMAP dataset (containing sparse/0/ and images/)
        output_path: Path where the undistorted dataset will be saved
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Undistorting COLMAP dataset from {input_path} to {output_path}")
    
    # Validate input paths
    sparse_path = input_path / "sparse" / "0"
    images_path = input_path / "images"
    
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse reconstruction not found at {sparse_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found at {images_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Running pycolmap undistort_images...")
    
    # Run image undistorter
    pycolmap.undistort_images(
        output_path=str(output_path),
        input_path=str(sparse_path),
        image_path=str(images_path),
        output_type="COLMAP"
    )
    
    # Fix directory structure: move sparse files to sparse/0/
    sparse_output = output_path / "sparse"
    sparse_0_output = output_path / "sparse" / "0"
    
    if sparse_output.exists() and not sparse_0_output.exists():
        print("Organizing sparse reconstruction files...")
        sparse_0_output.mkdir(parents=True, exist_ok=True)
        
        # Move all sparse files to sparse/0/
        for item in sparse_output.iterdir():
            if item.is_file() and item.suffix == ".bin":
                destination = sparse_0_output / item.name
                item.rename(destination)
                print(f"  Moved {item.name} to sparse/0/")
    
    print(f"\n✅ Undistortion complete!")
    print(f"📁 Undistorted dataset saved to: {output_path}")
    print(f"🚀 You can now run training with:")
    print(f"   gs.dataset.source_path={output_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python script/undistort_colmap.py <input_path> <output_path>")
        print("\nExample:")
        print("  python script/undistort_colmap.py data/colmap_otowa data/colmap_otowa_undistorted")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        undistort_colmap_dataset(input_path, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()