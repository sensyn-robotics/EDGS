import gc
import json
import os
import subprocess
import sys

from tqdm import tqdm


def convert_exr_to_png16_safe(exr_path, png_path, scale_factor=1000.0):
    """
    Convert EXR to PNG using subprocess to isolate segfaults.
    Returns (success, depth_stats) where depth_stats contains min/max depth info.
    """
    # Escape paths properly for subprocess
    exr_path_escaped = exr_path.replace("'", "\\'")
    png_path_escaped = png_path.replace("'", "\\'")

    conversion_script = f"""
import sys
import json
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
    
    # Calculate depth statistics before cleaning
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    original_min = float(np.min(valid_depth)) if len(valid_depth) > 0 else 0.0
    original_max = float(np.max(valid_depth)) if len(valid_depth) > 0 else 1.0
    
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0
    depth_scaled = (depth * {scale_factor}).clip(0, 65535).astype(np.uint16)
    iio.imwrite('{png_path_escaped}', depth_scaled)
    exr_file.close()
    
    # Output statistics as JSON
    stats = {{
        "original_min": original_min,
        "original_max": original_max,
        "scale_factor": {scale_factor}
    }}
    print("SUCCESS:" + json.dumps(stats))
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", conversion_script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )

        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            # Extract depth statistics from output
            success_line = [
                line
                for line in result.stdout.split("\n")
                if line.startswith("SUCCESS:")
            ][0]
            stats_json = success_line.replace("SUCCESS:", "")
            depth_stats = json.loads(stats_json)
            return True, depth_stats
        else:
            if result.stderr:
                print(
                    f"Subprocess stderr for {os.path.basename(exr_path)}: {result.stderr.strip()}"
                )
            return False, None

    except subprocess.TimeoutExpired:
        print(f"Timeout processing {os.path.basename(exr_path)}")
        return False, None
    except Exception as e:
        print(f"Subprocess error for {os.path.basename(exr_path)}: {e}")
        return False, None


def main(input_dir, output_dir, scale_factor):
    """
    Converts all .exr files in the input directory to .png files in the output directory.
    Also creates depth_params.json with conversion metadata.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    exr_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".exr")])

    if not exr_files:
        print(f"No .exr files found in {input_dir}")
        return

    print(f"Found {len(exr_files)} .exr files. Starting conversion...")

    failed_files = []
    successful_files = 0
    depth_params = {}

    for idx, exr_file in enumerate(tqdm(exr_files)):
        base_name = os.path.splitext(exr_file)[0]
        png_file = base_name + ".png"

        exr_full_path = os.path.join(input_dir, exr_file)
        png_full_path = os.path.join(output_dir, png_file)

        # Skip if already processed
        if os.path.exists(png_full_path):
            successful_files += 1
            # Still need to add to depth_params if not already there
            if base_name not in depth_params:
                depth_params[base_name] = {
                    "scale_factor": scale_factor,
                    "original_min": 0.0,  # Default values for existing files
                    "original_max": 65.535,
                    "format": "png",
                    "data_type": "uint16",
                }
            continue

        success, depth_stats = convert_exr_to_png16_safe(
            exr_full_path, png_full_path, scale_factor
        )

        if success and depth_stats:
            successful_files += 1
            # Store depth parameters for this file
            depth_params[base_name] = {
                "scale_factor": depth_stats["scale_factor"],
                "original_min": depth_stats["original_min"],
                "original_max": depth_stats["original_max"],
                "format": "png",
                "data_type": "uint16",
            }
        else:
            failed_files.append(exr_file)

        # Force garbage collection every 50 files
        if idx % 50 == 0:
            gc.collect()

    # Write depth_params.json
    depth_params_file = os.path.join(output_dir, "depth_params.json")
    with open(depth_params_file, "w") as f:
        json.dump(depth_params, f, indent=2)

    print(
        f"Conversion complete. {successful_files}/{len(exr_files)} files converted successfully."
    )
    print(
        f"Created depth_params.json with {len(depth_params)} entries at: {depth_params_file}"
    )

    if failed_files:
        print(f"Failed files ({len(failed_files)}): {failed_files[:10]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EXR depth maps to 16-bit PNG with metadata."
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
