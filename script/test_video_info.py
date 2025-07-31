#!/usr/bin/env python
"""Test video properties and extract a few frames manually"""

import cv2
import os
import sys

def test_video_properties(video_path):
    print(f"Testing video: {video_path}")
    
    if not os.path.exists(video_path):
        print("Video file does not exist!")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video!")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames (reported): {total_frames}")
    print(f"  FPS (reported): {fps}")
    print(f"  Resolution: {width}x{height}")
    
    # Try to extract first 10 frames manually
    print("\nExtracting first 10 frames...")
    frame_count = 0
    extracted = 0
    
    while frame_count < 1000 and extracted < 10:  # Limit to first 1000 attempts
        ret, frame = cap.read()
        if not ret:
            print(f"  Failed to read frame at position {frame_count}")
            break
            
        if frame_count % 100 == 0:  # Extract every 100th frame
            print(f"  Extracted frame {extracted+1} at position {frame_count}")
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nActual results:")
    print(f"  Successfully read {frame_count} frames")
    print(f"  Extracted {extracted} frames for testing")
    
    # Calculate realistic frame rate
    if frame_count > 0:
        # Estimate duration - assume this is a reasonable length video
        estimated_duration = 60  # seconds (guess)
        realistic_fps = frame_count / estimated_duration
        print(f"  Estimated realistic FPS: {realistic_fps:.1f}")

if __name__ == "__main__":
    video_path = "data/tower_by_drone.MP4"
    test_video_properties(video_path)