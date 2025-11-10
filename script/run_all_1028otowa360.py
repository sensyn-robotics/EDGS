#!/usr/bin/env python
# coding: utf-8

"""
Process all 360-degree videos from data/20251028_otowa360/ directory
and save results to outputs/20251028_otowa360_1m/.

Uses colmap_03_optimal_quality and train_04_medium_quality configs.
"""

import sys
import subprocess
from pathlib import Path
import glob

# Project root directory
project_root = Path(__file__).parent.parent.absolute()

# Configuration
INPUT_DIR = project_root / "data" / "20251028_otowa360"
OUTPUT_BASE_DIR = project_root / "outputs" / "20251028_otowa360_1m"
COLMAP_CONFIG = "colmap_03_optimal_quality"
TRAIN_CONFIG = "train_04_medium_quality"
MAX_RETRIES = 10  # Maximum number of retries for COLMAP failures

# Script to run
FIT_MODEL_SCRIPT = project_root / "script" / "fit_model_to_scene_full.py"


def get_video_files(input_dir):
    """Get all MP4 files in the input directory."""
    video_patterns = ["*.mp4", "*.MP4"]
    videos = []
    for pattern in video_patterns:
        videos.extend(glob.glob(str(input_dir / pattern)))
    return sorted(videos)


def get_output_name(video_path):
    """Generate output directory name from video filename."""
    # Remove extension and clean up the filename
    video_name = Path(video_path).stem
    return video_name


def process_video(video_path, output_dir, max_retries=MAX_RETRIES):
    """Process a single video file with automatic retry on failure."""
    print(f"\n{'='*80}")
    print(f"Processing: {Path(video_path).name}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        sys.executable,
        str(FIT_MODEL_SCRIPT),
        "--input", str(video_path),
        "--output_path", str(output_dir),
        "--colmap_config", COLMAP_CONFIG,
        "--config", TRAIN_CONFIG,
        "--360"  # Enable 360-degree mode
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    # Try processing with retries
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"\n[Retry {attempt}/{max_retries}] Retrying {Path(video_path).name}")
                print("(COLMAP often succeeds after retry due to memory issues)\n")

            subprocess.run(cmd, check=True)

            # Success message
            print(f"\n[SUCCESS] Processed: {Path(video_path).name}")
            if attempt > 1:
                print(f"          (Succeeded on attempt {attempt}/{max_retries})")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n[ATTEMPT {attempt}/{max_retries} FAILED] Return code: {e.returncode}")

            if attempt < max_retries:
                print(f"Will retry... ({max_retries - attempt} attempts remaining)")
            else:
                print(f"\n[FAILED] All {max_retries} attempts exhausted for {Path(video_path).name}")
                return False

    return False


def main():
    """Main processing loop."""
    # Check if input directory exists
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    # Get all video files
    video_files = get_video_files(INPUT_DIR)

    if not video_files:
        print(f"[ERROR] No MP4 files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) to process:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {Path(video).name}")
    print()

    # Create base output directory if it doesn't exist
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Process each video
    results = []
    for i, video_path in enumerate(video_files, 1):
        output_name = get_output_name(video_path)
        output_dir = OUTPUT_BASE_DIR / output_name

        print(f"\n[{i}/{len(video_files)}] Starting processing...")
        success = process_video(video_path, output_dir)
        results.append((Path(video_path).name, success))

    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    for video_name, success in results:
        status = "[SUCCESS]" if success else "[FAILED] "
        print(f"  {status}: {video_name}")

    print(f"\nTotal: {len(results)} videos")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
