#!/usr/bin/env python3
"""
Crop COLMAP scene to remove background points.
Keeps only points within a specified region or based on point statistics.
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


def filter_points_by_bbox(points3D, bbox_min, bbox_max):
    """Filter points that fall within bounding box."""
    filtered_ids = []
    for point_id, point in points3D.items():
        xyz = point.xyz
        if (bbox_min[0] <= xyz[0] <= bbox_max[0] and
            bbox_min[1] <= xyz[1] <= bbox_max[1] and
            bbox_min[2] <= xyz[2] <= bbox_max[2]):
            filtered_ids.append(point_id)
    return filtered_ids


def filter_points_by_center_radius(points3D, center, radius):
    """Filter points within radius from center."""
    filtered_ids = []
    center = np.array(center)
    for point_id, point in points3D.items():
        xyz = point.xyz
        if np.linalg.norm(xyz - center) <= radius:
            filtered_ids.append(point_id)
    return filtered_ids


def filter_points_by_percentile(points3D, percentile=95):
    """Remove outlier points beyond percentile distance from centroid."""
    # Get all point coordinates
    coords = np.array([p.xyz for p in points3D.values()])
    
    # Calculate centroid
    centroid = np.mean(coords, axis=0)
    
    # Calculate distances from centroid
    distances = np.linalg.norm(coords - centroid, axis=1)
    
    # Find percentile threshold
    threshold = np.percentile(distances, percentile)
    
    # Filter points
    filtered_ids = []
    for point_id, point in points3D.items():
        if np.linalg.norm(point.xyz - centroid) <= threshold:
            filtered_ids.append(point_id)
    
    return filtered_ids, centroid, threshold


def filter_points_by_reprojection_error(points3D, max_error=4.0):
    """Filter points with high reprojection error."""
    filtered_ids = []
    for point_id, point in points3D.items():
        if point.error <= max_error:
            filtered_ids.append(point_id)
    return filtered_ids


def filter_points_by_track_length(points3D, min_track_length=3):
    """Filter points seen in at least min_track_length images."""
    filtered_ids = []
    for point_id, point in points3D.items():
        if point.track.length() >= min_track_length:
            filtered_ids.append(point_id)
    return filtered_ids


def main():
    parser = argparse.ArgumentParser(description='Crop COLMAP scene to remove background')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input COLMAP sparse reconstruction')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output cropped COLMAP reconstruction')
    
    # Filtering methods
    parser.add_argument('--method', type=str, default='percentile',
                        choices=['bbox', 'radius', 'percentile', 'combined'],
                        help='Filtering method to use')
    
    # Bounding box parameters
    parser.add_argument('--bbox_min', type=float, nargs=3,
                        help='Minimum coordinates for bounding box (x y z)')
    parser.add_argument('--bbox_max', type=float, nargs=3,
                        help='Maximum coordinates for bounding box (x y z)')
    
    # Radius filtering parameters
    parser.add_argument('--center', type=float, nargs=3,
                        help='Center point for radius filtering (x y z)')
    parser.add_argument('--radius', type=float, default=50.0,
                        help='Radius for filtering')
    
    # Percentile filtering
    parser.add_argument('--percentile', type=float, default=95,
                        help='Percentile of points to keep (removes outliers)')
    
    # Quality filters
    parser.add_argument('--max_reproj_error', type=float, default=4.0,
                        help='Maximum reprojection error')
    parser.add_argument('--min_track_length', type=int, default=3,
                        help='Minimum track length (number of views)')
    
    args = parser.parse_args()
    
    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {args.input_path}")
    reconstruction = pycolmap.Reconstruction(args.input_path)
    
    print(f"Original scene statistics:")
    print(f"  Points: {len(reconstruction.points3D)}")
    print(f"  Images: {len(reconstruction.images)}")
    print(f"  Cameras: {len(reconstruction.cameras)}")
    
    # Get initial point IDs
    all_point_ids = set(reconstruction.points3D.keys())
    filtered_ids = all_point_ids.copy()
    
    # Apply filtering based on method
    if args.method == 'bbox':
        if not args.bbox_min or not args.bbox_max:
            print("Error: --bbox_min and --bbox_max required for bbox method")
            return
        filtered_ids = set(filter_points_by_bbox(
            reconstruction.points3D, args.bbox_min, args.bbox_max))
        
    elif args.method == 'radius':
        if not args.center:
            # Use centroid if center not specified
            coords = np.array([p.xyz for p in reconstruction.points3D.values()])
            center = np.mean(coords, axis=0)
            print(f"Using centroid as center: {center}")
        else:
            center = args.center
        filtered_ids = set(filter_points_by_center_radius(
            reconstruction.points3D, center, args.radius))
        
    elif args.method == 'percentile':
        filtered_ids, centroid, threshold = filter_points_by_percentile(
            reconstruction.points3D, args.percentile)
        filtered_ids = set(filtered_ids)
        print(f"Centroid: {centroid}")
        print(f"Distance threshold ({args.percentile}%): {threshold:.2f}")
        
    elif args.method == 'combined':
        # First apply percentile filtering
        filtered_ids, centroid, threshold = filter_points_by_percentile(
            reconstruction.points3D, args.percentile)
        filtered_ids = set(filtered_ids)
        print(f"After percentile filter: {len(filtered_ids)} points")
    
    # Always apply quality filters
    quality_filtered = set(filter_points_by_reprojection_error(
        reconstruction.points3D, args.max_reproj_error))
    filtered_ids &= quality_filtered
    print(f"After reprojection error filter: {len(filtered_ids)} points")
    
    track_filtered = set(filter_points_by_track_length(
        reconstruction.points3D, args.min_track_length))
    filtered_ids &= track_filtered
    print(f"After track length filter: {len(filtered_ids)} points")
    
    # Remove filtered points
    points_to_remove = all_point_ids - filtered_ids
    for point_id in points_to_remove:
        reconstruction.points3D.erase(point_id)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write filtered reconstruction
    print(f"\nWriting filtered reconstruction to {args.output_path}")
    reconstruction.write(args.output_path)
    
    print(f"\nFiltered scene statistics:")
    print(f"  Points: {len(reconstruction.points3D)} (removed {len(points_to_remove)})")
    print(f"  Images: {len(reconstruction.images)}")
    print(f"  Cameras: {len(reconstruction.cameras)}")
    
    # Print bounding box of remaining points
    if len(reconstruction.points3D) > 0:
        coords = np.array([p.xyz for p in reconstruction.points3D.values()])
        bbox_min = np.min(coords, axis=0)
        bbox_max = np.max(coords, axis=0)
        print(f"\nRemaining points bounding box:")
        print(f"  Min: {bbox_min}")
        print(f"  Max: {bbox_max}")
        print(f"  Size: {bbox_max - bbox_min}")


if __name__ == "__main__":
    main()