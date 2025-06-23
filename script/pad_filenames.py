import argparse
import glob
import os


def pad_filenames(base_dir):
    """
    Rename files to have zero-padded sequential numbering.

    Args:
        base_dir: Base directory containing input/ and exr/ subdirectories
    """

    # Rename .jpg files in input/ directory
    input_dir = os.path.join(base_dir, "input")
    if os.path.exists(input_dir):
        jpg_files = glob.glob(os.path.join(input_dir, "*.jpg"))
        print(f"Found {len(jpg_files)} JPG files in {input_dir}")

        for i, file_path in enumerate(sorted(jpg_files)):
            dir_name = os.path.dirname(file_path)
            new_name = f"{i:08d}.jpg"  # 8-digit zero-padding to match expected format
            new_path = os.path.join(dir_name, new_name)

            if file_path != new_path:  # Only rename if different
                os.rename(file_path, new_path)
                print(f"Renamed {os.path.basename(file_path)} -> {new_name}")
    else:
        print(f"Warning: {input_dir} directory not found")

    # Rename .exr files in exr/ directory
    exr_dir = os.path.join(base_dir, "exr")
    if os.path.exists(exr_dir):
        exr_files = glob.glob(os.path.join(exr_dir, "*.exr"))
        print(f"Found {len(exr_files)} EXR files in {exr_dir}")

        for i, file_path in enumerate(sorted(exr_files)):
            dir_name = os.path.dirname(file_path)
            new_name = f"{i:08d}.exr"  # 8-digit zero-padding
            new_path = os.path.join(dir_name, new_name)

            if file_path != new_path:  # Only rename if different
                os.rename(file_path, new_path)
                print(f"Renamed {os.path.basename(file_path)} -> {new_name}")
    else:
        print(f"Warning: {exr_dir} directory not found")

    print("File padding complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pad RGB and EXR filenames with zero-padded sequential numbering"
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
