#!/usr/bin/env python
# coding: utf-8

"""
Convert equirectangular 360° images to cubemap faces.

This module provides functionality to convert 360 degree equirectangular images
into cubemap face projections using ffmpeg's v360 filter.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def convert_equirectangular_to_cubemap(input_image_path, output_dir):
    """
    Convert an equirectangular 360 image to 5 cubemap face images.

    Note: Bottom face is excluded to avoid capturing the camera operator.

    Args:
        input_image_path: Path to equirectangular image
        output_dir: Directory to save cubemap face images

    Returns:
        List of paths to the 5 generated cubemap face images (front, right, back, left, top)
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Define the 5 cubemap faces with their respective ffmpeg parameters
    # Format: (suffix, yaw, pitch, roll)
    # Note: Bottom face is excluded as it often captures the camera operator
    faces = [
        ("front", 0, 0, 0),      # Front face
        ("right", -90, 0, 0),    # Right face
        ("back", 180, 0, 0),     # Back face
        ("left", 90, 0, 0),      # Left face
        ("top", 0, 90, 0),       # Top face (pitch=90 looks up)
    ]

    output_paths = []

    for face_name, yaw, pitch, roll in faces:
        output_path = os.path.join(output_dir, f"{basename}_{face_name}.png")

        # Build ffmpeg command for v360 filter
        # e:rectilinear converts equirectangular to rectilinear projection
        # h_fov and v_fov set the field of view to 90 degrees for cube face
        cmd = [
            "ffmpeg",
            "-i", input_image_path,
            "-vf", f"v360=e:rectilinear:h_fov=90:v_fov=90:yaw={yaw}:pitch={pitch}:roll={roll}",
            "-y",  # Overwrite output files
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            output_paths.append(output_path)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Warning: Failed to generate {face_name} face: {e.stderr}")
            continue

    return output_paths


def main():
    """Command-line interface for converting equirectangular images to cubemap."""
    parser = argparse.ArgumentParser(
        description="Convert 360° equirectangular images/videos to cubemap faces."
    )
    parser.add_argument(
        "input",
        help="Path to equirectangular image or video file"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save cubemap face images"
    )
    parser.add_argument(
        "--time",
        type=float,
        metavar="SECONDS",
        default=None,
        help="For video input: extract a single frame at this time (in seconds). Example: --time 5.0 extracts frame at 5 seconds"
    )
    parser.add_argument(
        "--fps",
        type=float,
        metavar="FPS",
        default=1.0,
        help="For video input: extract frames at this rate (frames per second). Default: 1.0"
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    # Detect if input is a video file
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    # For image files, disable fps-based extraction
    if not is_video and args.time is None:
        args.fps = None

    # Handle video input with single frame extraction
    if args.time is not None:
        print(f"📹 Extracting frame from video at {args.time} seconds...")

        # Create temporary directory for frame
        temp_dir = tempfile.mkdtemp()
        frame_path = os.path.join(temp_dir, "frame.png")

        # Extract frame using ffmpeg
        cmd = [
            "ffmpeg",
            "-ss", str(args.time),
            "-i", input_path,
            "-frames:v", "1",
            "-y",
            frame_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Frame extracted")
            input_path = frame_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting frame: {e.stderr}")
            return 1

    # Handle video input with fps-based extraction
    elif args.fps is not None:
        print(f"📹 Extracting frames from video at {args.fps} fps...")

        # Get video duration
        duration_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]

        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            print(f"   Video duration: {duration:.1f} seconds")
        except Exception:
            print("⚠️  Could not determine video duration, using 60 seconds as default")
            duration = 60.0

        # Calculate number of frames to extract
        num_frames = int(duration * args.fps)
        print(f"   Extracting approximately {num_frames} frames")

        # Extract frames at specified fps
        temp_dir = tempfile.mkdtemp()

        all_cubemap_faces = []
        for frame_idx in range(num_frames):
            time_pos = frame_idx / args.fps
            if time_pos >= duration:
                break

            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")

            cmd = [
                "ffmpeg",
                "-ss", str(time_pos),
                "-i", input_path,
                "-frames:v", "1",
                "-y",
                frame_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)

                # Convert this frame to cubemap
                frame_output_dir = os.path.join(output_dir, f"frame_{frame_idx:04d}")
                cubemap_faces = convert_equirectangular_to_cubemap(frame_path, frame_output_dir)
                all_cubemap_faces.extend(cubemap_faces)

                print(f"   ✅ Frame {frame_idx+1}/{num_frames} converted")

            except subprocess.CalledProcessError:
                print(f"   ⚠️  Warning: Failed to process frame at {time_pos:.1f}s")
                continue

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        print()
        print(f"✅ Successfully generated {len(all_cubemap_faces)} cubemap faces from {num_frames} frames")
        return 0

    # Convert to cubemap (for single image or single frame from video)
    print("🌐 Converting equirectangular image to cubemap faces...")
    print(f"   Input: {os.path.basename(input_path)}")
    print(f"   Output: {output_dir}")
    print()

    try:
        cubemap_faces = convert_equirectangular_to_cubemap(input_path, output_dir)

        print(f"✅ Successfully generated {len(cubemap_faces)} cubemap faces:")
        for face in cubemap_faces:
            print(f"   - {os.path.basename(face)}")

        # Clean up temp directory if we created one
        if args.time is not None:
            shutil.rmtree(temp_dir)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
