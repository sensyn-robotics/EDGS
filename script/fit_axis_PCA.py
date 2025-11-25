#!/usr/bin/env python3
"""
Align Gaussian Splatting PLY file axes using PCA.

This script reads a Gaussian Splatting .ply file and aligns its axes using
Principal Component Analysis (PCA). The output has:
- Z axis: largest variance direction (principal component 1)
- Y axis: second largest variance direction (principal component 2)
- X axis: smallest variance direction (principal component 3)
- Origin: center of gravity (centroid) of the point cloud
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z format)."""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def load_gaussian_ply(path):
    """Load a Gaussian Splatting PLY file."""
    plydata = PlyData.read(path)
    vertex = plydata.elements[0]

    # Extract xyz positions
    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"])
    ], axis=1)

    # Extract rotation quaternions (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    rotations = np.stack([
        np.asarray(vertex["rot_0"]),
        np.asarray(vertex["rot_1"]),
        np.asarray(vertex["rot_2"]),
        np.asarray(vertex["rot_3"])
    ], axis=1)

    return plydata, xyz, rotations


def compute_pca_transform(xyz):
    """
    Compute PCA transformation for the point cloud.

    Returns:
        centroid: center of gravity
        rotation_matrix: 3x3 matrix that transforms points to PCA-aligned coordinates
                        where Z=largest variance, Y=second, X=smallest
    """
    # Compute centroid (center of gravity)
    centroid = np.mean(xyz, axis=0)

    # Center the points
    centered = xyz - centroid

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest to smallest)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create rotation matrix:
    # - Column 0 (X): smallest variance direction (3rd eigenvector)
    # - Column 1 (Y): second largest variance direction (2nd eigenvector)
    # - Column 2 (Z): largest variance direction (1st eigenvector)
    rotation_matrix = np.column_stack([
        eigenvectors[:, 2],  # X = smallest variance
        eigenvectors[:, 1],  # Y = second largest
        eigenvectors[:, 0],  # Z = largest variance
    ])

    # Ensure right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] = -rotation_matrix[:, 0]

    print(f"PCA eigenvalues (variance): {eigenvalues}")
    print(f"Variance ratios: Z={eigenvalues[0]/sum(eigenvalues):.3f}, "
          f"Y={eigenvalues[1]/sum(eigenvalues):.3f}, "
          f"X={eigenvalues[2]/sum(eigenvalues):.3f}")

    return centroid, rotation_matrix


def transform_gaussian_ply(plydata, xyz, rotations, centroid, rotation_matrix):
    """
    Transform the Gaussian Splatting PLY data to PCA-aligned coordinates.

    Args:
        plydata: Original PlyData object
        xyz: Original xyz positions
        rotations: Original quaternions (w, x, y, z)
        centroid: PCA centroid to use as new origin
        rotation_matrix: 3x3 rotation matrix for PCA alignment

    Returns:
        New PlyData object with transformed data
    """
    # Transform positions: translate to centroid then rotate
    xyz_centered = xyz - centroid
    xyz_transformed = (rotation_matrix.T @ xyz_centered.T).T

    # Convert rotation matrix to quaternion
    rot_quat = rotation_matrix_to_quaternion(rotation_matrix.T)

    # Transform each Gaussian's rotation quaternion
    rotations_transformed = np.array([
        quaternion_multiply(rot_quat, q) for q in rotations
    ])

    # Normalize quaternions
    norms = np.linalg.norm(rotations_transformed, axis=1, keepdims=True)
    rotations_transformed = rotations_transformed / norms

    # Create new vertex data
    vertex = plydata.elements[0]
    property_names = [p.name for p in vertex.properties]

    # Build new structured array
    new_data = np.empty(len(vertex.data), dtype=vertex.data.dtype)

    # Copy all original data first
    for name in property_names:
        new_data[name] = vertex[name]

    # Update transformed values
    new_data['x'] = xyz_transformed[:, 0].astype(np.float32)
    new_data['y'] = xyz_transformed[:, 1].astype(np.float32)
    new_data['z'] = xyz_transformed[:, 2].astype(np.float32)
    new_data['rot_0'] = rotations_transformed[:, 0].astype(np.float32)
    new_data['rot_1'] = rotations_transformed[:, 1].astype(np.float32)
    new_data['rot_2'] = rotations_transformed[:, 2].astype(np.float32)
    new_data['rot_3'] = rotations_transformed[:, 3].astype(np.float32)

    # Create new PLY element and data
    new_element = PlyElement.describe(new_data, 'vertex')
    new_plydata = PlyData([new_element], text=plydata.text)

    return new_plydata


def main():
    parser = argparse.ArgumentParser(
        description="Align Gaussian Splatting PLY axes using PCA. "
                    "Output axes: Z=largest variance, Y=second, X=smallest. "
                    "Origin is set to the point cloud centroid."
    )
    parser.add_argument(
        "input",
        help="Path to input Gaussian Splatting .ply file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output .ply file. If not specified, appends '_pca' to input filename"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_pca")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load PLY file
    print("Loading Gaussian Splatting PLY file...")
    plydata, xyz, rotations = load_gaussian_ply(input_path)
    print(f"Loaded {len(xyz)} Gaussians")
    print()

    # Compute PCA transform
    print("Computing PCA transformation...")
    centroid, rotation_matrix = compute_pca_transform(xyz)
    print(f"Centroid (new origin): {centroid}")
    print(f"Rotation matrix:\n{rotation_matrix}")
    print()

    # Print original bounding box
    print("Original bounding box:")
    print(f"  X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"  Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
    print(f"  Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
    print()

    # Transform
    print("Transforming point cloud...")
    new_plydata = transform_gaussian_ply(plydata, xyz, rotations, centroid, rotation_matrix)

    # Print transformed bounding box
    new_xyz = np.stack([
        new_plydata.elements[0]['x'],
        new_plydata.elements[0]['y'],
        new_plydata.elements[0]['z']
    ], axis=1)
    print("Transformed bounding box:")
    print(f"  X: [{new_xyz[:, 0].min():.3f}, {new_xyz[:, 0].max():.3f}]")
    print(f"  Y: [{new_xyz[:, 1].min():.3f}, {new_xyz[:, 1].max():.3f}]")
    print(f"  Z: [{new_xyz[:, 2].min():.3f}, {new_xyz[:, 2].max():.3f}]")
    print()

    # Save
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_plydata.write(str(output_path))

    print("Done!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
