# convert_depth.py
import gc
import os

import imageio.v2 as imageio
import numpy as np
import pyexr
from tqdm import tqdm


def convert_exr_to_png16(exr_path, png_path, scale_factor=1000.0):
    """
    Reads a single-channel EXR depth file, scales it, and saves as a 16-bit PNG.
    """
    try:
        # Read EXR file
        exr_file = pyexr.open(exr_path)
        depth = exr_file.get(
            "R"
        )  # Depth is often in the 'R' channel for single-channel EXRs
        exr_file.close()

        # Remove invalid values
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0

        # Scale the depth and convert to 16-bit integer
        depth_scaled = (depth * scale_factor).clip(0, 65535).astype(np.uint16)

        # Squeeze to (H, W) if shape is (H, W, 1)
        if depth_scaled.ndim == 3 and depth_scaled.shape[2] == 1:
            depth_scaled = np.squeeze(depth_scaled, axis=2)

        # Save as 16-bit PNG
        imageio.imwrite(png_path, depth_scaled)

        # Explicitly delete arrays to free memory
        del depth
        del depth_scaled

    except Exception as e:
        print(f"Error processing {exr_path}: {e}")


def main(input_dir, output_dir, scale_factor):
    """
    Converts all .exr files in the input directory to .png files in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    exr_files = [f for f in os.listdir(input_dir) if f.endswith(".exr")]

    if not exr_files:
        print(f"No .exr files found in {input_dir}")
        return

    print(f"Found {len(exr_files)} .exr files. Starting conversion...")

    for exr_file in tqdm(exr_files):
        base_name = os.path.splitext(exr_file)[0]
        png_file = base_name + ".png"

        exr_full_path = os.path.join(input_dir, exr_file)
        png_full_path = os.path.join(output_dir, png_file)

        convert_exr_to_png16(exr_full_path, png_full_path, scale_factor)
        gc.collect()

    print("Conversion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EXR depth maps to 16-bit PNG."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing .exr files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save .png files."
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1000.0,
        help="Factor to scale depth values by (e.g., 1000 for meters to mm).",
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.scale_factor)
