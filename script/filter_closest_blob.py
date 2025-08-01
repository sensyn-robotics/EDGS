#!/usr/bin/env python3
"""
Filter COLMAP reconstruction to keep only the closest point cloud blob to cameras.
This removes disconnected background elements while preserving the main subject.
Method: Uses DBSCAN clustering to identify separate point cloud "blobs" and keeps only the closest ones to cameras.

  How it works:
  1. Clusters points into separate groups using spatial proximity (DBSCAN)
  2. Analyzes each cluster to find which is closest to camera trajectory
  3. Keeps entire closest cluster(s) while removing distant clusters completely

  Best for:
  - Scenes with clearly separated objects (e.g. tower + distant mountains)
  - When you want to remove entire disconnected regions
  - Complex scenes with multiple distinct structures

  Pros:
  - More intelligent - understands scene structure
  - Can remove entire background elements (mountains, buildings, etc.)
  - Preserves structural integrity of main object

  Cons:
  - More computationally intensive (requires clustering)
  - Memory hungry (got killed on your system)
  - May struggle if tower and background are connected

"""

import numpy as np
import argparse
from pathlib import Path
import sys
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import json
from datetime import datetime

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
        # Get camera center (position in world coordinates)
        try:
            # Try new pycolmap API
            if hasattr(image, 'cam_from_world'):
                R = image.cam_from_world.rotation.matrix()
                t = image.cam_from_world.translation
            else:
                # Try older API
                R = image.rotmat()
                t = image.tvec
            # Camera center is -R^T * t
            center = -R.T @ t
        except:
            # Fallback: try different methods
            try:
                # Alternative method
                center = image.projection_center()
                center = np.array(center)
            except:
                # Last resort - use translation vector as approximation
                try:
                    if hasattr(image, 'cam_from_world'):
                        center = -image.cam_from_world.translation
                    else:
                        center = -image.tvec
                except:
                    print(f"Warning: Could not extract center for image {image_id}")
                    continue
        
        camera_centers.append(center)
    return np.array(camera_centers)


def cluster_points(points3D, eps=2.0, min_samples=10):
    """
    Cluster 3D points using DBSCAN to identify separate blobs.
    
    Args:
        points3D: Dict of point_id -> point3D objects
        eps: Maximum distance between points in the same cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        labels: Array of cluster labels (-1 for noise)
        point_ids: Array of point IDs corresponding to labels
    """
    # Extract coordinates and IDs
    point_ids = []
    coords = []
    for pid, point in points3D.items():
        point_ids.append(pid)
        coords.append(point.xyz)
    
    coords = np.array(coords)
    point_ids = np.array(point_ids)
    
    # Cluster using DBSCAN
    print(f"Clustering {len(coords)} points with DBSCAN (eps={eps}, min_samples={min_samples})...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(coords)
    
    return labels, point_ids, coords


def find_closest_cluster(coords, labels, camera_centers):
    """
    Find the cluster that is closest to the camera positions.
    
    Args:
        coords: Array of 3D point coordinates
        labels: Array of cluster labels
        camera_centers: Array of camera center positions
    
    Returns:
        closest_cluster_id: ID of the closest cluster
    """
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise label
    
    min_avg_distance = float('inf')
    closest_cluster_id = None
    
    for cluster_id in unique_labels:
        # Get points in this cluster
        cluster_mask = labels == cluster_id
        cluster_points = coords[cluster_mask]
        
        # Calculate average distance from cameras to cluster
        cluster_tree = KDTree(cluster_points)
        distances = []
        
        for cam_center in camera_centers:
            # Find nearest point in cluster to this camera
            dist, _ = cluster_tree.query(cam_center)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        cluster_size = np.sum(cluster_mask)
        
        print(f"  Cluster {cluster_id}: {cluster_size} points, avg distance to cameras: {avg_distance:.2f}")
        
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            closest_cluster_id = cluster_id
    
    return closest_cluster_id


def analyze_clusters_by_distance(coords, labels, camera_centers):
    """Analyze each cluster's distance statistics from cameras."""
    unique_labels = sorted(set(labels))
    
    print("\nCluster Analysis:")
    print("-" * 80)
    print(f"{'Cluster':>8} {'Points':>10} {'Min Dist':>10} {'Avg Dist':>10} {'Max Dist':>10} {'Std Dev':>10}")
    print("-" * 80)
    
    cluster_stats = []
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            label = "Noise"
        else:
            label = str(cluster_id)
            
        # Get points in this cluster
        cluster_mask = labels == cluster_id
        cluster_points = coords[cluster_mask]
        
        if len(cluster_points) == 0:
            continue
            
        # Calculate distances from cameras to cluster
        cluster_tree = KDTree(cluster_points)
        all_distances = []
        
        for cam_center in camera_centers:
            # Find distances to all points in cluster
            dists, _ = cluster_tree.query(cam_center, k=min(10, len(cluster_points)))
            all_distances.extend(dists)
        
        min_dist = np.min(all_distances)
        avg_dist = np.mean(all_distances)
        max_dist = np.max(all_distances)
        std_dist = np.std(all_distances)
        
        cluster_stats.append({
            'id': int(cluster_id),  # Convert to int for JSON serialization
            'size': int(len(cluster_points)),  # Convert to int
            'min_dist': float(min_dist),  # Convert to float
            'avg_dist': float(avg_dist),  # Convert to float
            'max_dist': float(max_dist),  # Convert to float
            'std_dist': float(std_dist)  # Convert to float
        })
        
        print(f"{label:>8} {len(cluster_points):>10,} {min_dist:>10.2f} {avg_dist:>10.2f} {max_dist:>10.2f} {std_dist:>10.2f}")
    
    print("-" * 80)
    return cluster_stats


def adaptive_clustering(points3D, camera_centers, initial_eps=2.0):
    """
    Perform adaptive clustering to find good separation of blobs.
    """
    # Try different eps values to find good clustering
    eps_values = [initial_eps * factor for factor in [0.5, 1.0, 2.0, 4.0]]
    
    best_eps = initial_eps
    best_n_clusters = 0
    
    for eps in eps_values:
        labels, _, _ = cluster_points(points3D, eps=eps, min_samples=10)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  eps={eps:.1f}: {n_clusters} clusters found")
        
        # We want multiple clusters but not too many
        if 2 <= n_clusters <= 20:
            best_eps = eps
            best_n_clusters = n_clusters
            break
    
    if best_n_clusters < 2:
        print("Warning: Could not separate point cloud into multiple clusters.")
        print("The scene might be well-connected. Trying with smaller eps...")
        # Try with very small eps
        eps = initial_eps * 0.1
        labels, _, _ = cluster_points(points3D, eps=eps, min_samples=5)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            best_eps = eps
    
    return best_eps


def save_filter_log(output_path, args, original_stats, filtered_stats, clustering_details, clusters_kept):
    """Save filtering parameters and results to a log file."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "script": "filter_closest_blob.py",
        "parameters": {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "eps": args.eps,
            "min_samples": args.min_samples,
            "auto_eps": args.auto_eps,
            "keep_n_closest": args.keep_n_closest
        },
        "clustering_details": clustering_details,
        "clusters_kept": clusters_kept,
        "statistics": {
            "original": original_stats,
            "filtered": filtered_stats
        }
    }
    
    log_file = Path(output_path) / "filter_log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nFilter log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter COLMAP scene to keep only closest point cloud blob to cameras')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input COLMAP sparse reconstruction')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output filtered COLMAP reconstruction')
    parser.add_argument('--eps', type=float, default=2.0,
                        help='DBSCAN epsilon parameter (max distance between points in cluster)')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='DBSCAN minimum samples per cluster')
    parser.add_argument('--auto_eps', action='store_true',
                        help='Automatically determine good eps value')
    parser.add_argument('--keep_n_closest', type=int, default=1,
                        help='Number of closest clusters to keep')
    
    args = parser.parse_args()
    
    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {args.input_path}")
    reconstruction = pycolmap.Reconstruction(args.input_path)
    
    print(f"\nOriginal scene statistics:")
    print(f"  Points: {len(reconstruction.points3D):,}")
    print(f"  Images: {len(reconstruction.images)}")
    print(f"  Cameras: {len(reconstruction.cameras)}")
    
    # Store original statistics
    original_stats = {
        "points": len(reconstruction.points3D),
        "images": len(reconstruction.images),
        "cameras": len(reconstruction.cameras)
    }
    
    if len(reconstruction.points3D) == 0:
        print("Error: No 3D points in reconstruction")
        return
    
    # Get camera centers
    camera_centers = get_camera_centers(reconstruction)
    print(f"\nExtracted {len(camera_centers)} camera centers")
    
    # Determine eps if auto mode
    eps = args.eps
    if args.auto_eps:
        print("\nFinding optimal clustering parameters...")
        eps = adaptive_clustering(reconstruction.points3D, camera_centers, initial_eps=args.eps)
        print(f"Selected eps: {eps}")
    
    # Cluster points
    labels, point_ids, coords = cluster_points(
        reconstruction.points3D, eps=eps, min_samples=args.min_samples)
    
    # Count clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nClustering results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    if n_clusters == 0:
        print("Error: No clusters found. Try adjusting eps parameter.")
        return
    
    # Analyze clusters
    cluster_stats = analyze_clusters_by_distance(coords, labels, camera_centers)
    
    # Store clustering details
    clustering_details = {
        "eps_used": eps,
        "min_samples": args.min_samples,
        "total_clusters": n_clusters,
        "noise_points": n_noise,
        "cluster_stats": cluster_stats
    }
    
    # Find closest clusters
    valid_clusters = [c for c in cluster_stats if c['id'] != -1]
    valid_clusters.sort(key=lambda x: x['avg_dist'])
    
    clusters_to_keep = []
    clusters_kept_info = []
    for i in range(min(args.keep_n_closest, len(valid_clusters))):
        cluster_id = valid_clusters[i]['id']
        clusters_to_keep.append(cluster_id)
        clusters_kept_info.append({
            "cluster_id": int(cluster_id),
            "size": int(valid_clusters[i]['size']),
            "avg_distance": float(valid_clusters[i]['avg_dist'])
        })
        print(f"\nKeeping cluster {valid_clusters[i]['id']} "
              f"({valid_clusters[i]['size']:,} points, "
              f"avg distance: {valid_clusters[i]['avg_dist']:.2f})")
    
    # Filter points
    points_to_keep = set()
    for i, label in enumerate(labels):
        if label in clusters_to_keep:
            points_to_keep.add(point_ids[i])
    
    # Remove points not in closest cluster
    points_to_remove = set(reconstruction.points3D.keys()) - points_to_keep
    for point_id in points_to_remove:
        reconstruction.points3D.erase(point_id)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write filtered reconstruction
    print(f"\nWriting filtered reconstruction to {args.output_path}")
    reconstruction.write(args.output_path)
    
    print(f"\nFiltered scene statistics:")
    print(f"  Points: {len(reconstruction.points3D):,} (removed {len(points_to_remove):,})")
    print(f"  Images: {len(reconstruction.images)}")
    print(f"  Cameras: {len(reconstruction.cameras)}")
    
    # Collect filtered statistics
    filtered_stats = {
        "points": len(reconstruction.points3D),
        "points_removed": len(points_to_remove),
        "reduction_percentage": 100 * len(points_to_remove) / original_stats["points"] if original_stats["points"] > 0 else 0,
        "images": len(reconstruction.images),
        "cameras": len(reconstruction.cameras)
    }
    
    # Calculate bounding box of remaining points
    if len(reconstruction.points3D) > 0:
        remaining_coords = np.array([p.xyz for p in reconstruction.points3D.values()])
        bbox_min = np.min(remaining_coords, axis=0)
        bbox_max = np.max(remaining_coords, axis=0)
        bbox_size = bbox_max - bbox_min
        
        filtered_stats["bounding_box"] = {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "size": bbox_size.tolist()
        }
    
    # Save filter log
    save_filter_log(args.output_path, args, original_stats, filtered_stats, clustering_details, clusters_kept_info)


if __name__ == "__main__":
    main()