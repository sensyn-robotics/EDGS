#!/usr/bin/env python
# coding: utf-8

"""
Process videos to COLMAP scenes with frame extraction.

This module provides functionality to process video files (including 360° videos)
into COLMAP reconstruction scenes with uniform frame extraction.
"""

import os
import shutil
from pathlib import Path

import cv2

from source.utils_preprocess import run_colmap_on_scene
from source.convert_equirectangular_to_cubemap import convert_equirectangular_to_cubemap

# Face order for grouping 360 images by view direction
# This ensures sequential matching works well: all fronts → all rights → all backs → etc.
FACE_ORDER = ["front", "right", "back", "left", "top"]
FACES_PER_FRAME = len(FACE_ORDER)


def check_colmap_scene(directory_path, min_registered_images=2):
    """
    Check if a directory contains a valid COLMAP scene with sufficient registered images.

    Args:
        directory_path: Directory to check
        min_registered_images: Minimum number of registered images required (default: 2)

    Returns:
        bool: True if valid COLMAP scene exists with enough registered images
    """
    if not os.path.isdir(directory_path):
        return False

    # Check for required COLMAP directories/files
    sparse_dir = os.path.join(directory_path, "sparse")
    if not os.path.exists(sparse_dir):
        return False

    # Check for model folders (0, 1, etc.) or direct files
    model_dir = None

    # Check numbered model directories
    for i in range(10):  # Check first 10 possible model folders
        candidate_dir = os.path.join(sparse_dir, str(i))
        if os.path.exists(candidate_dir):
            # Check for essential COLMAP files
            cameras = os.path.join(candidate_dir, "cameras.bin")
            images = os.path.join(candidate_dir, "images.bin")

            cameras_txt = os.path.join(candidate_dir, "cameras.txt")
            images_txt = os.path.join(candidate_dir, "images.txt")

            if (os.path.exists(cameras) and os.path.exists(images)) or \
               (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
                model_dir = candidate_dir
                break

    # Also check for direct files in sparse directory
    if model_dir is None:
        cameras = os.path.join(sparse_dir, "cameras.bin")
        images = os.path.join(sparse_dir, "images.bin")
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")

        if (os.path.exists(cameras) and os.path.exists(images)) or \
           (os.path.exists(cameras_txt) and os.path.exists(images_txt)):
            model_dir = sparse_dir

    if model_dir is None:
        return False

    # Check the number of registered images
    try:
        import pycolmap
        # Try different pycolmap APIs (different versions have different APIs)
        if hasattr(pycolmap, 'Reconstruction'):
            reconstruction = pycolmap.Reconstruction(model_dir)
            num_registered = len(reconstruction.images)
        elif hasattr(pycolmap, 'SceneManager'):
            scene_manager = pycolmap.SceneManager(model_dir)
            scene_manager.load()
            num_registered = len(scene_manager.images)
        else:
            print(f"⚠️  Unknown pycolmap API, falling back to file check")
            return True

        if num_registered < min_registered_images:
            print(f"⚠️  COLMAP scene exists but only has {num_registered} registered images (need {min_registered_images}+)")
            return False
        print(f"✅ COLMAP scene has {num_registered} registered images")
        return True
    except Exception as e:
        print(f"⚠️  Could not verify COLMAP reconstruction: {e}")
        # Fall back to just checking file existence
        return True


def find_videos_in_directory(directory_path, extensions=('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI')):
    """
    Recursively find all video files in a directory and its subdirectories.

    Args:
        directory_path: Root directory to search
        extensions: Tuple of valid video file extensions

    Returns:
        List of video paths
    """
    videos = []
    directory_path = Path(directory_path)

    for ext in extensions:
        for video_path in directory_path.rglob(f'*{ext}'):
            videos.append(str(video_path))

    return sorted(videos)


def find_images_in_directory(directory_path, extensions=('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP')):
    """
    Find all image files in a directory.

    Args:
        directory_path: Directory to search
        extensions: Tuple of valid image file extensions

    Returns:
        List of image paths
    """
    images = []
    directory_path = Path(directory_path)

    # Only search in the immediate directory, not subdirectories
    for file_path in directory_path.glob('*'):
        if file_path.suffix in extensions:
            images.append(str(file_path))

    return sorted(images)


def get_min_registered_images(images_dir):
    """
    Calculate minimum required registered images based on total images.
    At least 30% of images should be registered, with minimum of 2.
    """
    if not os.path.exists(images_dir):
        return 2
    total_images = len(find_images_in_directory(images_dir))
    # Require at least 30% of images to be registered, minimum 2
    return max(2, int(total_images * 0.3))


def run_colmap_with_retry(output_path, colmap_cfg, images_dir, single_camera=False):
    """
    Run COLMAP with retry logic using progressively more lenient settings.

    Args:
        output_path: Directory containing images and where COLMAP output goes
        colmap_cfg: COLMAP configuration dictionary
        images_dir: Directory containing images
        single_camera: If True, force all images to share the same camera intrinsics

    Returns:
        bool: True if reconstruction was successful
    """
    import copy

    min_registered = get_min_registered_images(images_dir)
    total_images = len(find_images_in_directory(images_dir))

    # Define retry configurations with progressively more lenient settings
    retry_configs = [
        # First attempt: original config with video data type
        {
            'name': 'original with video matching',
            'config': colmap_cfg
        },
        # Second attempt: more lenient matching
        {
            'name': 'lenient matching',
            'config': {
                **colmap_cfg,
                'sift_matching': {
                    **colmap_cfg.get('sift_matching', {}),
                    'max_ratio': 0.9,
                    'max_distance': 0.8,
                    'cross_check': False,
                },
                'pipeline': {
                    **colmap_cfg.get('pipeline', {}),
                    'min_num_matches': 10,
                    'min_model_size': 3,
                },
                'mapper': {
                    **colmap_cfg.get('mapper', {}),
                    'init_min_num_inliers': 10,
                    'init_max_error': 8.0,
                    'init_min_tri_angle': 1.0,
                    'abs_pose_min_num_inliers': 8,
                    'abs_pose_max_error': 12.0,
                    'abs_pose_min_inlier_ratio': 0.1,
                }
            }
        },
        # Third attempt: very lenient for difficult scenes
        {
            'name': 'very lenient (difficult scenes)',
            'config': {
                **colmap_cfg,
                'sift_extraction': {
                    **colmap_cfg.get('sift_extraction', {}),
                    'max_num_features': 16384,  # More features
                    'peak_threshold': 0.004,    # Lower threshold = more features
                },
                'sift_matching': {
                    'max_ratio': 0.95,
                    'max_distance': 0.9,
                    'cross_check': False,
                    'max_num_matches': 65536,
                },
                'pipeline': {
                    **colmap_cfg.get('pipeline', {}),
                    'min_num_matches': 5,
                    'min_model_size': 2,
                },
                'mapper': {
                    'init_min_num_inliers': 8,
                    'init_max_error': 12.0,
                    'init_min_tri_angle': 0.5,
                    'abs_pose_min_num_inliers': 5,
                    'abs_pose_max_error': 16.0,
                    'abs_pose_min_inlier_ratio': 0.05,
                    'filter_max_reproj_error': 8.0,
                    'filter_min_tri_angle': 0.25,
                }
            }
        },
    ]

    for attempt, retry_info in enumerate(retry_configs):
        print(f"\n🔄 COLMAP attempt {attempt + 1}/{len(retry_configs)}: {retry_info['name']}")

        # Clear previous reconstruction if retrying
        if attempt > 0:
            sparse_dir = os.path.join(output_path, "sparse")
            db_path = os.path.join(output_path, "database.db")
            if os.path.exists(sparse_dir):
                shutil.rmtree(sparse_dir)
            if os.path.exists(db_path):
                os.remove(db_path)
            print("  🗑️  Cleared previous reconstruction data")

        try:
            run_colmap_on_scene(output_path, force_pinhole=True, colmap_config=retry_info['config'],
                              single_camera=single_camera)

            # Check if reconstruction was successful
            if check_colmap_scene(output_path, min_registered_images=min_registered):
                print(f"✅ COLMAP reconstruction successful with {retry_info['name']}")
                return True
            else:
                print(f"⚠️  Reconstruction quality insufficient, will retry with more lenient settings")

        except Exception as e:
            print(f"❌ COLMAP failed: {e}")
            if attempt < len(retry_configs) - 1:
                print("  Will retry with more lenient settings...")
            continue

    # All attempts failed - provide helpful error message
    print(f"\n❌ COLMAP reconstruction failed after {len(retry_configs)} attempts")
    print(f"   Total images: {total_images}, Required registered: {min_registered}")
    print("\n💡 Suggestions:")
    print("   1. The video may have insufficient camera movement or overlap")
    print("   2. The scene may lack texture (smooth/reflective surfaces)")
    print("   3. Try recording with slower camera movement")
    print("   4. Ensure good lighting and avoid motion blur")
    return False


def process_video_to_colmap_scene(video_path, output_path, colmap_cfg, is_360=False, fov_360=120, max_frames=None):
    """
    Process video(s) with uniform frame extraction and run COLMAP.

    Args:
        video_path: Path to video file or directory
        output_path: Directory to save COLMAP scene
        colmap_cfg: COLMAP configuration dictionary with preprocessing settings
        is_360: If True, process as 360 degree equirectangular video
        fov_360: Field of view for 360 perspective conversion (default 120°, ~25% overlap)
        max_frames: Maximum number of video frames to extract (None = no limit).
                    For 360 mode, total images = max_frames × 5 faces.

    Returns:
        scene_dir: Path to COLMAP scene directory
    """
    # Import here to avoid circular dependency
    from script.extract_frames_uniform import extract_frames_uniformly, get_video_duration_safe

    # Check if input is a directory or single video file
    is_directory = os.path.isdir(video_path)

    if is_directory:
        print(f"🎥 Processing videos from directory: {video_path}")
        videos = find_videos_in_directory(video_path)
        if not videos:
            raise ValueError(f"No video files found in {video_path}")
        print(f"📊 Found {len(videos)} video(s)")
    else:
        print(f"🎥 Processing single video: {video_path}")
        videos = [video_path]

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    images_dir = os.path.join(output_path, "images")

    # Check if images already exist
    if os.path.exists(images_dir):
        existing_images = find_images_in_directory(images_dir)
        if existing_images:
            print(f"✅ Found {len(existing_images)} existing images in {images_dir}, skipping extraction")

            # Calculate minimum required registered images
            min_registered = get_min_registered_images(images_dir)

            # Check if COLMAP has already been run AND has sufficient quality
            if check_colmap_scene(output_path, min_registered_images=min_registered):
                print(f"✅ COLMAP reconstruction already exists with sufficient registered images")
                return output_path
            else:
                print("🏗️  Running COLMAP reconstruction on existing images...")
                if run_colmap_with_retry(output_path, colmap_cfg, images_dir, single_camera=is_360):
                    print(f"🎉 COLMAP processing complete!")
                    return output_path
                else:
                    raise RuntimeError("COLMAP reconstruction failed - see suggestions above")

    os.makedirs(images_dir, exist_ok=True)

    # Process each video
    all_frame_paths = []
    frame_counter = 0

    # Calculate frames per video for multiple videos
    if is_directory and len(videos) > 1:
        total_target_frames = 500  # Total frames target for multiple videos
        frames_per_video = max(50, total_target_frames // len(videos))
    else:
        frames_per_video = None

    # Enforce max_total_images cap (default 1000 for 360 mode)
    max_total_images = colmap_cfg.get('processing_360', {}).get('max_total_images', 1000) if is_360 else None
    if is_360 and max_frames is None and max_total_images:
        max_frames = max_total_images // FACES_PER_FRAME
        print(f"📊 Max total images: {max_total_images} → max {max_frames} frames (× {FACES_PER_FRAME} faces)")

    for idx, video_file in enumerate(videos):
        print(f"\n🔄 Processing video {idx+1}/{len(videos)}: {os.path.basename(video_file)}")

        # Let extract_frames_uniformly handle all the video metadata checking
        # It has robust handling for corrupted metadata
        if frames_per_video:
            target_num_frames = frames_per_video
        else:
            # Get actual video duration and calculate frames based on target_fps
            duration = get_video_duration_safe(video_file)
            target_fps = colmap_cfg.get('preprocessing', {}).get('target_fps', 3.0)

            if duration and duration > 0:
                target_num_frames = int(target_fps * duration)
                print(f"  📊 Video duration: {duration:.1f}s, extracting {target_num_frames} frames at {target_fps} fps")
            else:
                target_num_frames = 300
                print(f"  ⚠️ Could not determine video duration, using default {target_num_frames} frames")

        # Apply max_frames cap
        if max_frames and target_num_frames > max_frames:
            print(f"  📊 Capping frames from {target_num_frames} to {max_frames}")
            target_num_frames = max_frames

        print(f"  🎯 Requesting {target_num_frames} frames")

        # Extract frames to a temporary directory
        temp_dir = os.path.join(output_path, f"temp_video_{idx}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            print("  🔄 Extracting frames uniformly...")
            max_image_size = colmap_cfg.get('preprocessing', {}).get('max_image_size', -1)
            frame_paths = extract_frames_uniformly(
                video_path=video_file,
                output_dir=temp_dir,
                num_frames=target_num_frames,
                max_size=max_image_size if max_image_size > 0 else 2048
            )

            # Process frames: convert to cubemap if 360 mode, otherwise just move
            if is_360:
                print(f"  🌐 Converting {len(frame_paths)} equirectangular frames to cubemap faces...")
                cubemap_temp_dir = os.path.join(output_path, f"temp_cubemap_{idx}")
                os.makedirs(cubemap_temp_dir, exist_ok=True)

                # Collect faces per frame for interleaved ordering
                frames_faces = []  # list of lists: [[front, right, back, left, top], ...]

                for frame_path in frame_paths:
                    cubemap_faces = convert_equirectangular_to_cubemap(frame_path, cubemap_temp_dir, fov=fov_360)
                    # Collect faces in consistent order for this frame
                    frame_face_paths = []
                    for face_name in FACE_ORDER:
                        if face_name in cubemap_faces:
                            frame_face_paths.append(cubemap_faces[face_name])
                    frames_faces.append(frame_face_paths)

                # Order by frame (interleaved): frame0_front, frame0_right, ..., frame0_top, frame1_front, ...
                # This enables sequential matching to cover:
                #   - Same-frame faces (within 5 neighbors)
                #   - Same-face temporal neighbors (5 apart)
                print(f"  📋 Ordering images by frame (interleaved) for sequential + loop closure matching...")
                for frame_face_paths in frames_faces:
                    for face_path in frame_face_paths:
                        new_filename = f"{frame_counter:08d}.png"
                        new_path = os.path.join(images_dir, new_filename)
                        shutil.move(face_path, new_path)
                        all_frame_paths.append(new_path)
                        frame_counter += 1

                # Clean up cubemap temp directory
                if os.path.exists(cubemap_temp_dir):
                    shutil.rmtree(cubemap_temp_dir)

                total_faces = sum(len(ff) for ff in frames_faces)
                print(f"  ✅ Converted {len(frame_paths)} frames to {total_faces} cubemap faces (excluding bottom)")
            else:
                # Move frames to combined directory with global numbering
                for frame_path in frame_paths:
                    new_filename = f"{frame_counter:08d}.png"
                    new_path = os.path.join(images_dir, new_filename)
                    shutil.move(frame_path, new_path)
                    all_frame_paths.append(new_path)
                    frame_counter += 1

                print(f"  ✅ Extracted {len(frame_paths)} frames")

        except Exception as e:
            print(f"  ❌ Failed to extract frames: {e}")
            continue
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    if not all_frame_paths:
        raise RuntimeError("No frames were extracted from any video")

    print(f"\n✅ Total frames extracted: {len(all_frame_paths)}")

    # Run COLMAP reconstruction with retry logic
    # For 360 mode, use single_camera=True to ensure identical intrinsics
    print("🏗️  Running COLMAP reconstruction...")
    if run_colmap_with_retry(output_path, colmap_cfg, images_dir, single_camera=is_360):
        print(f"🎉 COLMAP processing complete!")
        return output_path
    else:
        raise RuntimeError("COLMAP reconstruction failed - see suggestions above")
