#!/usr/bin/env python
# coding: utf-8

"""
Tree Diameter Estimator

Estimates tree trunk diameter using:
- Depth images from Gaussian Splatting rendering
- Tree segmentation results (COCO format)
- Camera focal length

Formula: tree_diameter = tree_width_pix * depth_m / focal_length_pix
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class TreeMeasurement:
    """Measurement result for a single tree."""
    tree_id: int
    image_id: int
    image_name: str
    category_id: int
    category_name: str
    trunk_width_pixels: float
    depth_meters: float
    focal_length_pixels: float
    diameter_meters: float
    confidence: float
    bbox: tuple  # (x, y, width, height)
    measurement_height_ratio: float  # Where on the tree the measurement was taken (0=bottom, 1=top)


class TreeDiameterEstimator:
    """
    Estimates tree trunk diameters from depth images and segmentation masks.

    Expected directory structure:
        scene_path/
            cameras.json              # Camera parameters with focal lengths
            train/ours_XXXXX/depth/   # Depth maps as .npy files
            *.json                    # Tree segmentation in COCO format (auto-detected)
    """

    # Tree category IDs (from COCO format segmentation)
    TREE_CATEGORIES = {'hinoki': 1, 'sugi': 2, 'broad-leaf': 3}

    def __init__(
        self,
        scene_path: str,
        segmentation_file: Optional[str] = None,
        depth_dir: Optional[str] = None,
        measurement_height_ratio: float = 0.8,
        min_tree_height_pixels: int = 100,
    ):
        """
        Initialize the tree diameter estimator.

        Args:
            scene_path: Path to the scene directory
            segmentation_file: Path to segmentation JSON (auto-detected if None)
            depth_dir: Path to depth directory (auto-detected if None)
            measurement_height_ratio: Where to measure trunk width (0=bottom, 1=top).
                                      Default 0.8 means 80% down from top (near trunk base)
            min_tree_height_pixels: Minimum tree height in pixels to consider
        """
        self.scene_path = Path(scene_path)
        self.measurement_height_ratio = measurement_height_ratio
        self.min_tree_height_pixels = min_tree_height_pixels

        # Load camera parameters
        self.cameras = self._load_cameras()

        # Load segmentation data
        self.segmentation_file = segmentation_file or self._find_segmentation_file()
        self.segmentation = self._load_segmentation()

        # Set depth directory
        self.depth_dir = Path(depth_dir) if depth_dir else self._find_depth_dir()

        # Build lookup maps
        self._build_lookups()

    def _load_cameras(self) -> dict:
        """Load camera parameters from cameras.json."""
        cameras_path = self.scene_path / "cameras.json"
        if not cameras_path.exists():
            raise FileNotFoundError(f"cameras.json not found at {cameras_path}")

        with open(cameras_path) as f:
            cameras_list = json.load(f)

        # Index by image name for easy lookup
        cameras = {}
        for cam in cameras_list:
            img_name = cam['img_name']
            cameras[img_name] = cam
        return cameras

    def _is_coco_segmentation(self, file_path: Path) -> bool:
        """Check if a JSON file is in COCO segmentation format."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            # COCO format requires these keys with annotations containing segmentation
            if not isinstance(data, dict):
                return False
            if not all(key in data for key in ['images', 'annotations', 'categories']):
                return False
            # Check that annotations have segmentation data
            if data['annotations'] and 'segmentation' in data['annotations'][0]:
                return True
            return False
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return False

    def _find_segmentation_file(self) -> Path:
        """Find segmentation JSON file in the scene directory."""
        # First, try files with 'inferences' in the name (legacy behavior)
        for f in self.scene_path.iterdir():
            if f.suffix == '.json' and 'inferences' in f.name.lower():
                return f

        # Otherwise, check all JSON files for COCO segmentation format
        for f in self.scene_path.iterdir():
            if f.suffix == '.json' and f.name != 'cameras.json':
                if self._is_coco_segmentation(f):
                    return f

        raise FileNotFoundError(f"No COCO segmentation JSON found in {self.scene_path}")

    def _load_segmentation(self) -> dict:
        """Load COCO-format segmentation data."""
        seg_path = Path(self.segmentation_file)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        with open(seg_path) as f:
            return json.load(f)

    def _find_depth_dir(self) -> Path:
        """Find depth directory in train outputs."""
        train_dir = self.scene_path / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"train directory not found at {train_dir}")

        # Find the latest ours_XXXXX directory
        ours_dirs = sorted([d for d in train_dir.iterdir() if d.name.startswith("ours_")])
        if not ours_dirs:
            raise FileNotFoundError(f"No ours_* directory found in {train_dir}")

        depth_dir = ours_dirs[-1] / "depth"
        if not depth_dir.exists():
            raise FileNotFoundError(f"depth directory not found at {depth_dir}")

        return depth_dir

    def _build_lookups(self):
        """Build lookup dictionaries for images and categories."""
        # Image ID to image info
        self.image_lookup = {img['id']: img for img in self.segmentation['images']}

        # Category ID to category name
        self.category_lookup = {}
        for cat in self.segmentation['categories']:
            if cat['id'] not in self.category_lookup:
                self.category_lookup[cat['id']] = cat['name']

        # Build camera name to index mapping (for depth file lookup)
        self.camera_index = {cam['img_name']: idx for idx, cam in enumerate(self.cameras.values())}

    def _polygon_to_mask(self, segmentation: list, width: int, height: int) -> np.ndarray:
        """Convert COCO polygon segmentation to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def _measure_trunk_width(
        self,
        mask: np.ndarray,
        bbox: tuple,
        measurement_ratio: float
    ) -> tuple[float, int]:
        """
        Measure trunk width at a specific height ratio within the bounding box.

        Args:
            mask: Binary segmentation mask
            bbox: Bounding box (x, y, width, height)
            measurement_ratio: Where to measure (0=top of bbox, 1=bottom)

        Returns:
            (trunk_width_pixels, measurement_row)
        """
        x, y, w, h = [int(v) for v in bbox]

        # Calculate measurement row (within the bounding box)
        measurement_row = int(y + h * measurement_ratio)
        measurement_row = min(measurement_row, mask.shape[0] - 1)

        # Get the mask row and find the extent of the tree
        row_mask = mask[measurement_row, :]

        if row_mask.sum() == 0:
            # No pixels at this row, search nearby rows
            for offset in range(1, 20):
                for direction in [-1, 1]:
                    test_row = measurement_row + offset * direction
                    if 0 <= test_row < mask.shape[0]:
                        row_mask = mask[test_row, :]
                        if row_mask.sum() > 0:
                            measurement_row = test_row
                            break
                if row_mask.sum() > 0:
                    break

        if row_mask.sum() == 0:
            return 0.0, measurement_row

        # Find leftmost and rightmost pixels
        nonzero_cols = np.where(row_mask > 0)[0]
        if len(nonzero_cols) == 0:
            return 0.0, measurement_row

        trunk_width = nonzero_cols[-1] - nonzero_cols[0] + 1
        return float(trunk_width), measurement_row

    def _get_depth_at_region(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        measurement_row: int,
        trunk_width: float
    ) -> float:
        """
        Get representative depth value at the measurement region.

        Args:
            depth_map: Depth image
            mask: Binary segmentation mask
            measurement_row: Row where trunk width was measured
            trunk_width: Width of trunk in pixels

        Returns:
            Depth in meters (or the raw depth value if scale unknown)
        """
        # Define a small region around the measurement row
        row_range = 5  # +/- 5 rows
        row_start = max(0, measurement_row - row_range)
        row_end = min(depth_map.shape[0], measurement_row + row_range + 1)

        # Get depth values within the mask region
        region_mask = mask[row_start:row_end, :]
        region_depth = depth_map[row_start:row_end, :]

        masked_depth = region_depth[region_mask > 0]

        if len(masked_depth) == 0:
            return 0.0

        # Use median depth to be robust to outliers
        return float(np.median(masked_depth))

    def _find_matching_camera(self, image_name: str) -> Optional[dict]:
        """Find camera parameters for an image, handling name variations."""
        # Direct match
        if image_name in self.cameras:
            return self.cameras[image_name]

        # Try without extension
        base_name = os.path.splitext(image_name)[0]
        for cam_name, cam in self.cameras.items():
            if os.path.splitext(cam_name)[0] == base_name:
                return cam

        # Try matching by number
        try:
            img_num = int(''.join(filter(str.isdigit, base_name)))
            for cam_name, cam in self.cameras.items():
                cam_num = int(''.join(filter(str.isdigit, os.path.splitext(cam_name)[0])))
                if cam_num == img_num:
                    return cam
        except ValueError:
            pass

        return None

    def _find_depth_file(self, image_name: str) -> Optional[Path]:
        """Find depth file corresponding to an image.

        Depth files are numbered 00000.npy, 00001.npy, etc., corresponding
        to the order of cameras in cameras.json.
        """
        # Check if this image has a corresponding camera
        if image_name not in self.camera_index:
            return None

        idx = self.camera_index[image_name]
        depth_file = self.depth_dir / f"{idx:05d}.npy"

        if depth_file.exists():
            return depth_file

        return None

    def estimate_diameters(
        self,
        image_ids: Optional[list[int]] = None,
        category_ids: Optional[list[int]] = None,
        min_confidence: float = 0.5,
    ) -> list[TreeMeasurement]:
        """
        Estimate tree diameters for all trees in the segmentation.

        Args:
            image_ids: Specific image IDs to process (all if None)
            category_ids: Specific category IDs to process (all tree types if None)
            min_confidence: Minimum segmentation confidence score

        Returns:
            List of TreeMeasurement objects
        """
        measurements = []

        for annotation in self.segmentation['annotations']:
            # Filter by confidence
            confidence = annotation.get('score', 1.0)
            if confidence < min_confidence:
                continue

            # Filter by category
            if category_ids and annotation['category_id'] not in category_ids:
                continue

            # Filter by image
            if image_ids and annotation['image_id'] not in image_ids:
                continue

            # Get image info
            image_info = self.image_lookup.get(annotation['image_id'])
            if not image_info:
                continue

            image_name = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']

            # Get camera parameters (only process images that have cameras)
            camera = self._find_matching_camera(image_name)
            if not camera:
                # This is expected - segmentation may cover more images than training set
                continue

            # Use fx as the focal length (horizontal)
            focal_length = camera['fx']

            # Find and load depth file
            depth_file = self._find_depth_file(image_name)
            if not depth_file:
                # This can happen if depth was not rendered for this image
                continue

            depth_map = np.load(depth_file)
            if depth_map.ndim == 3:
                depth_map = depth_map[0]  # Remove channel dimension if present

            # Resize depth map if needed (depth might be at different resolution)
            if depth_map.shape != (img_height, img_width):
                depth_map = cv2.resize(depth_map, (img_width, img_height))

            # Get bounding box
            bbox = tuple(annotation['bbox'])  # (x, y, width, height)

            # Skip small trees
            if bbox[3] < self.min_tree_height_pixels:
                continue

            # Create segmentation mask
            mask = self._polygon_to_mask(
                annotation['segmentation'],
                img_width,
                img_height
            )

            # Measure trunk width
            trunk_width, measurement_row = self._measure_trunk_width(
                mask, bbox, self.measurement_height_ratio
            )

            if trunk_width <= 0:
                continue

            # Get depth at measurement location
            depth = self._get_depth_at_region(depth_map, mask, measurement_row, trunk_width)

            if depth <= 0:
                continue

            # Calculate diameter using the formula
            # diameter = trunk_width_pix * depth_m / focal_length_pix
            diameter = trunk_width * depth / focal_length

            # Create measurement result
            measurement = TreeMeasurement(
                tree_id=annotation['id'],
                image_id=annotation['image_id'],
                image_name=image_name,
                category_id=annotation['category_id'],
                category_name=self.category_lookup.get(annotation['category_id'], 'unknown'),
                trunk_width_pixels=trunk_width,
                depth_meters=depth,
                focal_length_pixels=focal_length,
                diameter_meters=diameter,
                confidence=confidence,
                bbox=bbox,
                measurement_height_ratio=self.measurement_height_ratio,
            )
            measurements.append(measurement)

        return measurements

    def estimate_diameter_single(
        self,
        depth_map: np.ndarray,
        segmentation_mask: np.ndarray,
        focal_length_pixels: float,
        bbox: Optional[tuple] = None,
    ) -> Optional[float]:
        """
        Estimate diameter for a single tree given raw inputs.

        Args:
            depth_map: 2D depth array in meters
            segmentation_mask: Binary mask of the tree
            focal_length_pixels: Camera focal length in pixels
            bbox: Optional bounding box (x, y, w, h), computed from mask if None

        Returns:
            Estimated diameter in meters, or None if measurement failed
        """
        if bbox is None:
            # Compute bounding box from mask
            rows = np.any(segmentation_mask, axis=1)
            cols = np.any(segmentation_mask, axis=0)
            if not rows.any() or not cols.any():
                return None
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = (cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)

        # Measure trunk width
        trunk_width, measurement_row = self._measure_trunk_width(
            segmentation_mask, bbox, self.measurement_height_ratio
        )

        if trunk_width <= 0:
            return None

        # Get depth
        depth = self._get_depth_at_region(depth_map, segmentation_mask, measurement_row, trunk_width)

        if depth <= 0:
            return None

        # Calculate diameter
        return trunk_width * depth / focal_length_pixels


def estimate_tree_diameters(
    scene_path: str,
    output_file: Optional[str] = None,
    measurement_height_ratio: float = 0.8,
    min_confidence: float = 0.5,
) -> list[TreeMeasurement]:
    """
    Convenience function to estimate tree diameters for a scene.

    Args:
        scene_path: Path to the scene directory
        output_file: Optional JSON file to save results
        measurement_height_ratio: Where to measure (0.8 = 80% down from top)
        min_confidence: Minimum segmentation confidence

    Returns:
        List of TreeMeasurement objects
    """
    estimator = TreeDiameterEstimator(
        scene_path=scene_path,
        measurement_height_ratio=measurement_height_ratio,
    )

    measurements = estimator.estimate_diameters(min_confidence=min_confidence)

    if output_file:
        # Convert to serializable format
        results = []
        for m in measurements:
            results.append({
                'tree_id': m.tree_id,
                'image_id': m.image_id,
                'image_name': m.image_name,
                'category_id': m.category_id,
                'category_name': m.category_name,
                'trunk_width_pixels': m.trunk_width_pixels,
                'depth_meters': m.depth_meters,
                'focal_length_pixels': m.focal_length_pixels,
                'diameter_meters': m.diameter_meters,
                'confidence': m.confidence,
                'bbox': list(m.bbox),
                'measurement_height_ratio': m.measurement_height_ratio,
            })

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} measurements to {output_file}")

    return measurements


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate tree trunk diameters")
    parser.add_argument("scene_path", help="Path to the scene directory")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=0.8,
        help="Measurement height ratio (0=top, 1=bottom, default: 0.8)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum segmentation confidence (default: 0.5)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results"
    )

    args = parser.parse_args()

    measurements = estimate_tree_diameters(
        scene_path=args.scene_path,
        output_file=args.output,
        measurement_height_ratio=args.height_ratio,
        min_confidence=args.min_confidence,
    )

    # Print summary
    print(f"\nProcessed {len(measurements)} tree measurements")

    if measurements:
        diameters = [m.diameter_meters for m in measurements]
        print(f"Diameter range: {min(diameters):.3f}m - {max(diameters):.3f}m")
        print(f"Mean diameter: {np.mean(diameters):.3f}m")
        print(f"Median diameter: {np.median(diameters):.3f}m")

        # Group by category
        by_category = {}
        for m in measurements:
            cat = m.category_name
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(m.diameter_meters)

        print("\nBy tree type:")
        for cat, diams in sorted(by_category.items()):
            print(f"  {cat}: {len(diams)} trees, mean diameter {np.mean(diams):.3f}m")

        if args.verbose:
            print("\nDetailed measurements:")
            for m in measurements:
                print(f"  Tree {m.tree_id} ({m.category_name}): "
                      f"diameter={m.diameter_meters:.3f}m, "
                      f"depth={m.depth_meters:.2f}m, "
                      f"width={m.trunk_width_pixels:.0f}px")
