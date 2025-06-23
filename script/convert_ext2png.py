import gc
import os
import subprocess
import sys

import imageio.v3 as iio
import Imath
import numpy as np
import OpenEXR
from tqdm import tqdm


def convert_exr_to_png16_safe(exr_path, png_path, scale_factor=1000.0):
    """
    Convert EXR to PNG using subprocess to isolate segfaults.
    """
    # Escape paths properly for subprocess
    exr_path_escaped = exr_path.replace("'", "\\'")
    png_path_escaped = png_path.replace("'", "\\'")

    conversion_script = f"""
import sys
import os
try:
    import imageio.v3 as iio
    import Imath
    import numpy as np
    import OpenEXR
    
    exr_file = OpenEXR.InputFile('{exr_path_escaped}')
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth = np.frombuffer(exr_file.channel("R", FLOAT), dtype=np.float32)
    depth = np.reshape(depth, (size[1], size[0]))
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0
    depth_scaled = (depth * {scale_factor}).clip(0, 65535).astype(np.uint16)
    iio.imwrite('{png_path_escaped}', depth_scaled)
    exr_file.close()
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

    try:
        # Run in completely isolated subprocess
        result = subprocess.run(
            [sys.executable, "-c", conversion_script],
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout
            cwd=os.getcwd(),  # Ensure proper working directory
        )

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return True
        else:
            # Print both stdout and stderr for debugging
            if result.stderr:
                print(
                    f"Subprocess stderr for {os.path.basename(exr_path)}: {result.stderr.strip()}"
                )
            if result.stdout and "ERROR:" in result.stdout:
                print(
                    f"Subprocess stdout for {os.path.basename(exr_path)}: {result.stdout.strip()}"
                )
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout processing {os.path.basename(exr_path)}")
        return False
    except Exception as e:
        print(f"Subprocess error for {os.path.basename(exr_path)}: {e}")
        return False


def convert_exr_to_png16(exr_path, png_path, scale_factor=1000.0):
    """
    Original in-process conversion (kept as fallback).
    """
    exr_file = None
    try:
        exr_file = OpenEXR.InputFile(exr_path)
        dw = exr_file.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        depth = np.frombuffer(exr_file.channel("R", FLOAT), dtype=np.float32)
        depth = np.reshape(depth, (size[1], size[0]))

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0

        depth_scaled = (depth * scale_factor).clip(0, 65535).astype(np.uint16)
        iio.imwrite(png_path, depth_scaled)
        del depth
        del depth_scaled
        return True
    except Exception as e:
        print(f"Error processing {exr_path}: {e}")
        return False
    finally:
        if exr_file is not None:
            exr_file.close()
            del exr_file


def main(
    input_dir, output_dir, scale_factor, use_subprocess=False
):  # Changed default to False
    """
    Converts all .exr files in the input directory to .png files in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    exr_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".exr")])

    if not exr_files:
        print(f"No .exr files found in {input_dir}")
        return

    print(f"Found {len(exr_files)} .exr files. Starting conversion...")
    print(f"Using {'subprocess' if use_subprocess else 'direct'} method...")

    failed_files = []
    successful_files = 0

    for idx, exr_file in enumerate(tqdm(exr_files)):
        base_name = os.path.splitext(exr_file)[0]
        png_file = base_name + ".png"

        exr_full_path = os.path.join(input_dir, exr_file)
        png_full_path = os.path.join(output_dir, png_file)

        # Skip if already processed
        if os.path.exists(png_full_path):
            successful_files += 1
            continue

        # Choose conversion method
        if use_subprocess:
            success = convert_exr_to_png16_safe(
                exr_full_path, png_full_path, scale_factor
            )
        else:
            success = convert_exr_to_png16(exr_full_path, png_full_path, scale_factor)

        if success:
            successful_files += 1
        else:
            failed_files.append(exr_file)

        # Force garbage collection every 50 files
        if idx % 50 == 0:
            gc.collect()

    print(
        f"Conversion complete. {successful_files}/{len(exr_files)} files converted successfully."
    )
    if failed_files:
        print(f"Failed files ({len(failed_files)}): {failed_files[:10]}...")


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
    parser.add_argument(
        "--use_subprocess",
        action="store_true",
        help="Use subprocess to isolate segfaults (recommended).",
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.scale_factor, args.use_subprocess)
