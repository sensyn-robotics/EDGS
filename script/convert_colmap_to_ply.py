#!/usr/bin/env python3
"""Convert COLMAP sparse reconstruction to PLY point cloud."""

import os
import sys
import numpy as np
import pycolmap

def write_ply(filename, points, colors=None):
    """Write a PLY file with points and optional colors."""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(len(points)):
            if colors is not None:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
                f.write(f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")

def convert_colmap_to_ply(sparse_path, output_ply_path):
    """Convert COLMAP sparse reconstruction to PLY format."""
    
    # Read the reconstruction
    print(f"Reading COLMAP reconstruction from: {sparse_path}")
    reconstruction = pycolmap.Reconstruction(sparse_path)
    
    # Extract points and colors
    points_list = []
    colors_list = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points_list.append(point3D.xyz)
        colors_list.append(point3D.color)
    
    if len(points_list) == 0:
        print("No 3D points found in the reconstruction!")
        return False
    
    points = np.array(points_list)
    colors = np.array(colors_list)
    
    print(f"Found {len(points)} 3D points")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Write PLY file
    print(f"Writing PLY file to: {output_ply_path}")
    write_ply(output_ply_path, points, colors)
    
    print(f"Successfully wrote {len(points)} points to {output_ply_path}")
    return True

if __name__ == "__main__":
    # Default paths
    sparse_path = "outputs/otowa360/sparse/0"
    output_ply_path = "outputs/otowa360/point_cloud.ply"
    
    # Check if custom paths provided
    if len(sys.argv) > 1:
        sparse_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_ply_path = sys.argv[2]
    
    # Convert
    success = convert_colmap_to_ply(sparse_path, output_ply_path)
    
    if not success:
        sys.exit(1)