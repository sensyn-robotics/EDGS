import argparse
import glob
import os
import re


def pad_files_in_directory(directory, file_extension):
    """
    Add zero-padding to filenames in a specific directory while preserving original numbers.

    Args:
        directory: Directory path containing files to pad
        file_extension: File extension to process (e.g., 'jpg', 'exr')

    Returns:
        Number of files processed
    """
    if not os.path.exists(directory):
        print(f"Warning: {directory} directory not found")
        return 0

    files = glob.glob(os.path.join(directory, f"*.{file_extension}"))
    print(f"Found {len(files)} {file_extension.upper()} files in {directory}")

    processed_count = 0
    for file_path in files:
        old_name = os.path.basename(file_path)
        # Extract number from filename
        match = re.search(r"(\d+)", old_name)
        if match:
            number = int(match.group(1))
            new_name = f"{number:08d}.{file_extension}"  # Pad the original number
            new_path = os.path.join(directory, new_name)

            if file_path != new_path:
                os.rename(file_path, new_path)
                print(f"  Renamed {old_name} -> {new_name}")
                processed_count += 1
        else:
            print(f"  Skipping {old_name} - no number found")

    print(f"{file_extension.upper()} files padded while preserving original numbers")
    return processed_count


def pad_filenames(base_dir):
    """
    Add zero-padding to record3D filenames while preserving original numbers.
    Processes rgb/*.jpg and depth/*.exr files.

    Args:
        base_dir: Base directory containing rgb/ and depth/ subdirectories
    """

    print(f"Processing record3D files in: {base_dir}")

    # Process RGB files
    rgb_dir = os.path.join(base_dir, "rgb")
    rgb_count = pad_files_in_directory(rgb_dir, "jpg")

    # Process depth files
    depth_dir = os.path.join(base_dir, "depth")
    depth_count = pad_files_in_directory(depth_dir, "exr")

    print("\nFile padding complete!")
    print(f"✓ Processed {rgb_count} RGB files and {depth_count} depth files")
    print("✓ Original file numbers preserved - RGB/depth correspondence maintained")

    print("\nNext steps:")
    print(
        f"1. Run: python script/convert_ext2png.py --input_dir {base_dir}/depth/ --output_dir {base_dir}/depth_png/"
    )
    print(f"2. Run: python script/fit_model_to_scene_full.py --input_path {base_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pad record3D RGB and depth filenames with zero-padding while preserving original numbers"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing rgb/ and depth/ subdirectories",
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist")
        exit(1)

    pad_filenames(args.base_dir)
