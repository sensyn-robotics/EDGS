import argparse
import glob
import os
import re


def pad_filenames(base_dir):
    """
    Add zero-padding to filenames while preserving original numbers.
    This maintains correspondence between RGB images and depth files.

    Args:
        base_dir: Base directory containing input/ and exr/ subdirectories
    """

    print(f"Processing files in: {base_dir}")

    # Step 1: Pad .jpg files in input/ directory
    input_dir = os.path.join(base_dir, "input")
    if os.path.exists(input_dir):
        jpg_files = glob.glob(os.path.join(input_dir, "*.jpg"))
        print(f"Found {len(jpg_files)} JPG files in {input_dir}")

        for file_path in jpg_files:
            old_name = os.path.basename(file_path)
            # Extract number from filename
            match = re.search(r"(\d+)", old_name)
            if match:
                number = int(match.group(1))
                new_name = f"{number:08d}.jpg"  # Pad the original number
                new_path = os.path.join(input_dir, new_name)

                if file_path != new_path:
                    os.rename(file_path, new_path)
                    print(f"  Renamed {old_name} -> {new_name}")

        print("JPG files padded while preserving original numbers")
    else:
        print(f"Warning: {input_dir} directory not found")

    # Step 2: Pad .exr files in exr/ directory
    exr_dir = os.path.join(base_dir, "exr")
    if os.path.exists(exr_dir):
        exr_files = glob.glob(os.path.join(exr_dir, "*.exr"))
        print(f"Found {len(exr_files)} EXR files in {exr_dir}")

        for file_path in exr_files:
            old_name = os.path.basename(file_path)
            # Extract number from filename
            match = re.search(r"(\d+)", old_name)
            if match:
                number = int(match.group(1))
                new_name = f"{number:08d}.exr"  # Pad the original number
                new_path = os.path.join(exr_dir, new_name)

                if file_path != new_path:
                    os.rename(file_path, new_path)
                    print(f"  Renamed {old_name} -> {new_name}")

        print("EXR files padded while preserving original numbers")
    else:
        print(f"Warning: {exr_dir} directory not found")

    # Step 3: Clean up existing depth directory if it exists
    depth_dir = os.path.join(base_dir, "depth")
    if os.path.exists(depth_dir):
        print(f"Removing existing depth directory to force regeneration: {depth_dir}")
        import shutil

        shutil.rmtree(depth_dir)

    print("File padding complete!")
    print("✓ Original file numbers preserved - RGB/depth correspondence maintained")
    print("\nNext steps:")
    print(
        f"1. Run: python script/convert_ext2png.py --input_dir {base_dir}/exr/ --output_dir {base_dir}/depth/"
    )
    print(f"2. Run: python script/fit_model_to_scene_full.py --input_path {base_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pad RGB and EXR filenames with zero-padding while preserving original numbers"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing input/ and exr/ subdirectories",
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist")
        exit(1)

    pad_filenames(args.base_dir)
