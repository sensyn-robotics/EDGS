#!/usr/bin/env python3
"""
Simple distance-based filtering for COLMAP reconstruction.
Keeps points within a certain distance from camera trajectory.
Method: Uses simple distance thresholds from camera positions to filter points.

  How it works:
  1. Calculates distances from each point to camera trajectory
  2. Applies threshold to keep closest X% of points or points within radius
  3. Removes points individually based on distance only

  Best for:
  - Memory-efficient filtering
  - When main subject is consistently closer to cameras
  - Quick and simple background removal

  Pros:
  - Fast and memory efficient ✅
  - Simple and reliable
  - Worked successfully on your tower scene
  - Multiple filtering options (percentile, radius, quality)

  Cons:
  - Less intelligent - doesn't understand scene structure
  - May create "holes" in objects if parts are far from cameras
  - Can't distinguish between connected structures

"""

import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    import pycolmap
except ImportError:
    print("Error: pycolmap not found. Please install it first.")
    sys.exit(1)


def get_camera_centers(reconstruction):
    """Extract all camera centers from the reconstruction."""
    camera_centers = []
    for image_id, image in reconstruction.images.items():
        try:
            # Try projection_center first (most direct)
            center = np.array(image.projection_center())
        except:
            try:
                # Try alternative methods
                if hasattr(image, 'tvec'):
                    center = -image.tvec  # Approximation
                else:
                    continue
            except:
                print(f"Warning: Could not extract center for image {image_id}")
                continue
        
        camera_centers.append(center)
    return np.array(camera_centers)


def compute_camera_trajectory_bounds(camera_centers, expansion_factor=2.0):
    """Compute bounding region around camera trajectory."""
    # Find trajectory center and extent
    center = np.mean(camera_centers, axis=0)
    
    # Calculate distances from center
    distances = np.linalg.norm(camera_centers - center, axis=1)
    max_cam_distance = np.max(distances)
    
    # Expand the region
    trajectory_radius = max_cam_distance * expansion_factor
    
    return center, trajectory_radius


def filter_points_by_trajectory_distance(points3D, camera_centers, max_distance_factor=3.0):
    """
    Filter points based on distance from camera trajectory.
    Keeps points within max_distance_factor * camera_trajectory_radius
    """
    trajectory_center, trajectory_radius = compute_camera_trajectory_bounds(camera_centers)
    max_distance = trajectory_radius * max_distance_factor
    
    print(f"Camera trajectory center: {trajectory_center}")
    print(f"Camera trajectory radius: {trajectory_radius:.2f}")
    print(f"Filtering points beyond distance: {max_distance:.2f}")
    
    filtered_ids = []
    distances = []
    
    for point_id, point in points3D.items():
        xyz = point.xyz
        dist = np.linalg.norm(xyz - trajectory_center)
        distances.append(dist)
        
        if dist <= max_distance:
            filtered_ids.append(point_id)
    
    distances = np.array(distances)
    print(f"Point distances - Min: {np.min(distances):.2f}, "
          f"Max: {np.max(distances):.2f}, "
          f"Mean: {np.mean(distances):.2f}")
    
    return filtered_ids


def filter_points_by_percentile_distance(points3D, camera_centers, percentile=90):
    """
    Filter points by percentile distance from cameras.
    Keeps closest percentile% of points to camera centers.
    """
    print(f"Filtering to keep closest {percentile}% of points to cameras")
    
    all_distances = []
    point_distances = {}
    
    for point_id, point in points3D.items():
        xyz = point.xyz
        # Find minimum distance to any camera
        min_dist = np.min(np.linalg.norm(camera_centers - xyz, axis=1))
        all_distances.append(min_dist)
        point_distances[point_id] = min_dist
    
    # Find distance threshold
    threshold = np.percentile(all_distances, percentile)
    print(f"Distance threshold ({percentile}%): {threshold:.2f}")
    
    # Filter points
    filtered_ids = []
    for point_id, dist in point_distances.items():
        if dist <= threshold:
            filtered_ids.append(point_id)
    
    return filtered_ids


def get_point_quality_stats(points3D):
    """Analyze point quality statistics."""
    errors = []
    track_lengths = []
    
    for point in points3D.values():
        errors.append(point.error)
        track_lengths.append(point.track.length())
    
    errors = np.array(errors)
    track_lengths = np.array(track_lengths)
    
    print(f"\nPoint quality statistics:")
    print(f"  Reprojection errors - Min: {np.min(errors):.3f}, Max: {np.max(errors):.3f}, Mean: {np.mean(errors):.3f}")
    print(f"  Track lengths - Min: {np.min(track_lengths)}, Max: {np.max(track_lengths)}, Mean: {np.mean(track_lengths):.1f}")
    
    return errors, track_lengths


def main():
    parser = argparse.ArgumentParser(
        description='Filter COLMAP scene by distance from camera trajectory')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input COLMAP sparse reconstruction')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output filtered COLMAP reconstruction')
    
    # Filtering methods
    parser.add_argument('--method', type=str, default='percentile',
                        choices=['trajectory', 'percentile', 'quality'],
                        help='Filtering method')
    
    # Parameters
    parser.add_argument('--distance_factor', type=float, default=3.0,
                        help='Distance factor for trajectory-based filtering')
    parser.add_argument('--percentile', type=float, default=85,
                        help='Percentile of points to keep (based on camera distance)')
    parser.add_argument('--max_error', type=float, default=4.0,
                        help='Maximum reprojection error')
    parser.add_argument('--min_track_length', type=int, default=3,
                        help='Minimum track length')
    
    args = parser.parse_args()
    
    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {args.input_path}")
    reconstruction = pycolmap.Reconstruction(args.input_path)
    
    print(f"\nOriginal scene statistics:")
    print(f"  Points: {len(reconstruction.points3D):,}")
    print(f"  Images: {len(reconstruction.images)}")
    print(f"  Cameras: {len(reconstruction.cameras)}")
    
    if len(reconstruction.points3D) == 0:
        print("Error: No 3D points in reconstruction")
        return
    
    # Analyze point quality
    get_point_quality_stats(reconstruction.points3D)
    
    # Get camera centers
    camera_centers = get_camera_centers(reconstruction)
    if len(camera_centers) == 0:
        print("Error: Could not extract camera centers")
        return
    
    print(f"\nExtracted {len(camera_centers)} camera centers")
    
    # Filter points based on method
    if args.method == 'trajectory':
        filtered_ids = filter_points_by_trajectory_distance(
            reconstruction.points3D, camera_centers, args.distance_factor)
    elif args.method == 'percentile':
        filtered_ids = filter_points_by_percentile_distance(
            reconstruction.points3D, camera_centers, args.percentile)
    elif args.method == 'quality':
        # Filter by quality metrics only
        filtered_ids = []
        for point_id, point in reconstruction.points3D.items():
            if (point.error <= args.max_error and 
                point.track.length() >= args.min_track_length):
                filtered_ids.append(point_id)
    
    filtered_ids = set(filtered_ids)
    
    # Remove filtered points
    all_point_ids = set(reconstruction.points3D.keys())
    points_to_remove = all_point_ids - filtered_ids
    
    for point_id in points_to_remove:
        try:
            # Try different removal methods
            if hasattr(reconstruction.points3D, 'erase'):
                reconstruction.points3D.erase(point_id)
            elif hasattr(reconstruction.points3D, 'pop'):
                reconstruction.points3D.pop(point_id)
            else:
                del reconstruction.points3D[point_id]
        except:
            pass  # Point might already be removed
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write filtered reconstruction
    print(f"\nWriting filtered reconstruction to {args.output_path}")
    reconstruction.write(args.output_path)
    
    print(f"\nFiltered scene statistics:")
    print(f"  Points: {len(reconstruction.points3D):,} (removed {len(points_to_remove):,})")
    print(f"  Reduction: {100 * len(points_to_remove) / len(all_point_ids):.1f}%")
    
    # Print bounding box of remaining points
    if len(reconstruction.points3D) > 0:
        coords = np.array([p.xyz for p in reconstruction.points3D.values()])
        bbox_min = np.min(coords, axis=0)
        bbox_max = np.max(coords, axis=0)
        bbox_size = bbox_max - bbox_min
        
        print(f"\nRemaining points bounding box:")
        print(f"  Min: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
        print(f"  Max: [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
        print(f"  Size: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")


if __name__ == "__main__":
    main()