#!/usr/bin/env python3
"""
Uniform frame extraction script for sampling frames evenly across entire video duration.
This script ensures frames are sampled from beginning to end of the video.
"""

import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def extract_frames_with_imageio(video_path, output_dir, num_frames, max_size):
    """
    Extract frames using imageio with uniform sampling.
    """
    print("📚 Using imageio for frame extraction...")
    
    reader = imageio.get_reader(video_path)
    try:
        # Get total frame count
        total_frames = reader.count_frames()
        print(f"📊 Total frames available: {total_frames}")
        
        if total_frames <= 1:
            raise RuntimeError(f"Video only contains {total_frames} frames")
        
        # Calculate frame indices for uniform sampling
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
            print(f"Video has only {total_frames} frames, extracting all")
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            print(f"Sampling {num_frames} frames uniformly from {total_frames} total frames")
        
        extracted_paths = []
        
        with tqdm(total=len(frame_indices), desc="Extracting with imageio", unit="frame") as pbar:
            for i, frame_idx in enumerate(frame_indices):
                try:
                    frame = reader.get_data(frame_idx)
                    
                    # Resize if needed
                    if max_size > 0:
                        h, w = frame.shape[:2]
                        max_dim = max(h, w)
                        if max_dim > max_size:
                            scale = max_size / max_dim
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Save frame
                    output_filename = f"{i:08d}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    success = cv2.imwrite(output_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        extracted_paths.append(output_path)
                    else:
                        print(f"Failed to save frame {i}")
                    
                except Exception as e:
                    print(f"Failed to extract frame {frame_idx}: {e}")
                
                pbar.update(1)
        
        print(f"✅ Successfully extracted {len(extracted_paths)} frames using imageio")
        
        if len(extracted_paths) == 0:
            raise RuntimeError("No frames extracted with imageio")
            
        return extracted_paths
        
    finally:
        reader.close()


def extract_frames_with_ffmpeg(video_path, output_dir, num_frames, max_size):
    """
    Extract frames using ffmpeg with uniform sampling across video duration.
    """
    print("🎬 Using ffmpeg for frame extraction...")
    
    # First, get video duration using ffprobe
    try:
        # Get video info using ffprobe
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True, check=True)
        
        duration = float(result.stdout.strip())
        print(f"📊 Video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Calculate time intervals for uniform sampling
        time_interval = duration / num_frames
        print(f"🎯 Extracting {num_frames} frames at {time_interval:.2f}s intervals")
        
        extracted_paths = []
        
        # Extract frames at specific time intervals
        with tqdm(total=num_frames, desc="Extracting with ffmpeg", unit="frame") as pbar:
            for i in range(num_frames):
                time_pos = i * time_interval
                output_filename = f"{i:08d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Build ffmpeg command for extracting single frame at specific time
                cmd = [
                    'ffmpeg', '-y', '-ss', str(time_pos), '-i', video_path, 
                    '-vframes', '1', '-q:v', '2'
                ]
                
                # Add scaling if needed
                if max_size > 0:
                    cmd.extend(['-vf', f'scale=min({max_size}\\,iw):min({max_size}\\,ih):force_original_aspect_ratio=decrease'])
                
                cmd.append(output_path)
                
                # Run ffmpeg command
                try:
                    subprocess.run(cmd, capture_output=True, check=True)
                    extracted_paths.append(output_path)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to extract frame at {time_pos:.2f}s: {e}")
                
                pbar.update(1)
        
        print(f"✅ Successfully extracted {len(extracted_paths)} frames using ffmpeg")
        
        if len(extracted_paths) == 0:
            raise RuntimeError("No frames extracted with ffmpeg")
            
        return extracted_paths
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"ffmpeg extraction failed: {e}")


def extract_frames_sequential_with_skipping(video_path, output_dir, num_frames, max_size):
    """
    For videos with corrupted metadata, read sequentially and sample frames across duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    print("📖 Reading video sequentially to determine actual length...")
    
    # First pass: count actual readable frames by reading through video
    actual_frames = []
    frame_count = 0
    sample_every = 1000  # Sample every 1000 frames to estimate length
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_every == 0:
            actual_frames.append(frame_count)
        
        frame_count += 1
        
        # Show progress periodically
        if frame_count % 10000 == 0:
            print(f"  Read {frame_count} frames...")
    
    cap.release()
    
    print(f"📊 Found {frame_count} actual readable frames")
    
    if frame_count == 0:
        raise RuntimeError("No frames could be read from video")
    
    # Calculate frame indices for uniform sampling
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
        print(f"Video has only {frame_count} frames, will extract all")
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        print(f"Sampling {num_frames} frames uniformly from {frame_count} actual frames")
        print(f"Frame indices: {frame_indices[0]} to {frame_indices[-1]}")
    
    # Second pass: extract selected frames
    cap = cv2.VideoCapture(video_path)
    extracted_paths = []
    current_frame = 0
    frame_idx_set = set(frame_indices)
    
    print("🎯 Extracting selected frames...")
    with tqdm(total=len(frame_indices), desc="Extracting frames", unit="frame") as pbar:
        while current_frame < frame_count and len(extracted_paths) < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in frame_idx_set:
                # Resize frame if needed
                if max_size > 0:
                    h, w = frame.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > max_size:
                        scale = max_size / max_dim
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save frame
                frame_filename = f"{len(extracted_paths):08d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    extracted_paths.append(frame_path)
                    pbar.update(1)
            
            current_frame += 1
    
    cap.release()
    
    if len(extracted_paths) == 0:
        raise RuntimeError("No frames were successfully extracted!")
    
    return extracted_paths

def extract_frames_uniformly(video_path, output_dir, num_frames=800, max_size=1024):
    """
    Extract frames uniformly distributed across the entire video duration.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (distributed evenly)
        max_size: Maximum dimension for extracted frames
    
    Returns:
        List of extracted frame paths
    """
    print(f"Starting uniform frame extraction from: {video_path}")
    print(f"Target frames: {num_frames}, Max size: {max_size}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"  Resolution: {width}x{height}")
    
    # Handle videos with corrupted metadata
    if fps > 1000 or total_frames > 1000000:
        print(f"⚠️ Video metadata appears corrupted (fps={fps:.1f}, frames={total_frames})")
        
        # Try imageio first for corrupted videos, then ffmpeg, then OpenCV
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if IMAGEIO_AVAILABLE:
            try:
                print("🔧 Trying imageio extraction...")
                return extract_frames_with_imageio(video_path, output_dir, num_frames, max_size)
            except Exception as e:
                print(f"❌ imageio failed: {e}")
        
        try:
            print("🔧 Trying ffmpeg extraction...")
            return extract_frames_with_ffmpeg(video_path, output_dir, num_frames, max_size)
        except Exception as e:
            print(f"❌ ffmpeg failed: {e}")
            
        print("🔧 Falling back to sequential reading...")
        return extract_frames_sequential_with_skipping(video_path, output_dir, num_frames, max_size)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame indices to extract uniformly across video
    if total_frames <= num_frames:
        # If video has fewer frames than requested, take all frames
        frame_indices = list(range(total_frames))
        print(f"Video has only {total_frames} frames, extracting all")
    else:
        # Calculate uniform sampling across entire video duration
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        print(f"Sampling {num_frames} frames uniformly from {total_frames} total frames")
        print(f"Frame indices range: {frame_indices[0]} to {frame_indices[-1]}")
        print(f"Average interval: {(frame_indices[-1] - frame_indices[0]) / (len(frame_indices) - 1):.1f} frames")
    
    extracted_paths = []
    failed_extractions = 0
    
    print("Extracting frames...")
    with tqdm(total=len(frame_indices), desc="Extracting frames", unit="frame") as pbar:
        for i, frame_idx in enumerate(frame_indices):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                failed_extractions += 1
                pbar.update(1)
                continue
            
            # Resize frame if needed
            if max_size > 0:
                h, w = frame.shape[:2]
                max_dim = max(h, w)
                if max_dim > max_size:
                    scale = max_size / max_dim
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Save frame
            frame_filename = f"{i:08d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                extracted_paths.append(frame_path)
            else:
                failed_extractions += 1
                print(f"Failed to save frame {i} to {frame_path}")
            
            pbar.update(1)
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Successfully extracted: {len(extracted_paths)} frames")
    print(f"  Failed extractions: {failed_extractions}")
    print(f"  Output directory: {output_dir}")
    
    if len(extracted_paths) == 0:
        raise RuntimeError("No frames were successfully extracted!")
    
    # Verify we got frames from across the video duration
    first_frame_time = frame_indices[0] / fps if fps > 0 else 0
    last_frame_time = frame_indices[-1] / fps if fps > 0 else 0
    print(f"  Time span: {first_frame_time:.1f}s to {last_frame_time:.1f}s ({last_frame_time/60:.1f} minutes)")
    
    return extracted_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames uniformly from video")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("output_dir", help="Directory to save extracted frames")
    parser.add_argument("--num_frames", type=int, default=800, help="Number of frames to extract")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum frame dimension")
    
    args = parser.parse_args()
    
    extract_frames_uniformly(args.video_path, args.output_dir, args.num_frames, args.max_size)