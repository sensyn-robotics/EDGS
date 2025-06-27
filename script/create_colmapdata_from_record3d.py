import argparse
import json
import logging
import os
import subprocess
import sys

import imageio.v3 as iio  # For reading EXR and PNG
import numpy as np
import pycolmap  # COLMAP .binファイル書き込みに必要
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def validate_exr_file(file_path):
    """
    Validate if an EXR file is readable and contains valid depth data.

    Args:
        file_path: Path to the EXR file

    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Very small files are likely corrupted
        if os.path.getsize(file_path) < 100:
            return False

        # Use the same safe reading approach as read_exr_depth_safe
        # but just for validation
        success, depth_array = read_exr_depth_safe(file_path)
        if not success or depth_array is None:
            return False

        # Check if it contains ANY valid depth data (not ALL zeros/nans)
        valid_pixels = np.isfinite(depth_array) & (depth_array > 0)
        if np.any(valid_pixels):  # Changed from np.all to np.any
            return True
        else:
            return False

    except Exception:
        return False


def read_exr_depth_safe(exr_path):
    """
    Safely read EXR depth file using subprocess to avoid segfaults.
    Returns (success, depth_array) where depth_array is numpy array or None.
    """
    # Escape paths properly for subprocess
    exr_path_escaped = exr_path.replace("'", "\\'")

    reading_script = f"""
import sys
import json
import base64
try:
    import Imath
    import numpy as np
    import OpenEXR
    
    exr_file = OpenEXR.InputFile('{exr_path_escaped}')
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth = np.frombuffer(exr_file.channel("R", FLOAT), dtype=np.float32)
    depth = np.reshape(depth, (size[1], size[0]))
    
    # Clean the depth data
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0
    
    exr_file.close()
    
    # Convert to bytes and encode as base64 for safe transmission
    depth_bytes = depth.tobytes()
    depth_b64 = base64.b64encode(depth_bytes).decode('ascii')
    
    # Output shape and data
    result = {{
        "shape": depth.shape,
        "dtype": str(depth.dtype),
        "data": depth_b64
    }}
    print("SUCCESS:" + json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", reading_script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )

        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            # Extract depth data from output
            success_line = [
                line
                for line in result.stdout.split("\\n")
                if line.startswith("SUCCESS:")
            ][0]
            result_json = json.loads(success_line.replace("SUCCESS:", ""))

            # Reconstruct numpy array from base64 data
            import base64

            depth_bytes = base64.b64decode(result_json["data"])
            depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
            depth_array = depth_array.reshape(result_json["shape"])

            return True, depth_array
        else:
            if result.stderr:
                logger.error(
                    f"EXR read error for {os.path.basename(exr_path)}: {result.stderr.strip()}"
                )
            return False, None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout reading {os.path.basename(exr_path)}")
        return False, None
    except Exception as e:
        logger.error(f"EXR read subprocess error for {os.path.basename(exr_path)}: {e}")
        return False, None


def convert_record3d_to_colmap(
    record3d_path, colmap_output_path, num_points_to_sample=1000
):
    logger.info(
        f"Converting Record3D data to COLMAP format: {record3d_path} -> {colmap_output_path}"
    )

    # 1. crate COLMAP directories
    images_dir = os.path.join(colmap_output_path, "images")
    sparse_dir = os.path.join(colmap_output_path, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # 2. read metadata.json exported by RecordeD
    metadata_path = os.path.join(record3d_path, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"cannot find metadata.json. check metadata_path: {metadata_path}")
        return False
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rgb_width = metadata["w"]
    rgb_height = metadata["h"]
    # prepare COLMAP Camera object: model, width, height, params. [ref](https://github.com/colmap/colmap/tree/main/python)
    # metadata.jsonの`perFrameIntrinsicCoeffs`provides each frame's `[fx, fy, cx, cy]`
    # So we use initial frame's one.
    first_intrinsics = metadata["perFrameIntrinsicCoeffs"][0]
    # COLMAPのPINHOLE expects fx, fy, cx, cy.
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=rgb_width,
        height=rgb_height,
        params=np.array(
            [
                first_intrinsics[0],  # fx
                first_intrinsics[1],  # fy
                first_intrinsics[2],  # cx
                first_intrinsics[3],  # cy
            ],
            dtype=np.float64,
        ),
    )
    camera.camera_id = 1  # Explicitly set the camera ID
    colmap_cameras = {1: camera}

    colmap_images = {}
    colmap_points3D = {}
    point_id_counter = 1  # unique id for 3D points.

    logger.info("start processing for each frame...")
    for i, frame_pose_data in enumerate(metadata["poses"]):
        # Record3D provides `poses` as `[qx, qy, qz, qw, tx, ty, tz]` 。
        # This represents camera to world coordinate transformation. (world_from_camera)。
        # P_world = R_wc @ P_cam + t_wc

        # COLMAP's images.bin expects world to camera coordinate transformation (camera_from_world).
        # P_cam = R_cw @ P_world + t_cw
        # And, R_cw = R_wc.T, and t_cw = -R_cw @ t_wc

        # extract Record3D q and t.
        quat_record3d_xyzw = np.array(frame_pose_data[:4])
        t_world_from_cam = np.array(frame_pose_data[4:])

        # (scipy.spatial.transform.Rotation expects [x,y,z,w])
        rot_world_from_cam = R.from_quat(quat_record3d_xyzw)

        # get world_to_camera (camera_from_world) by rotating the pose.
        rot_cam_from_world = rot_world_from_cam.inv()
        t_cam_from_world = -rot_cam_from_world.apply(t_world_from_cam)
        # scipy.Rotation.apply(v) computes R @ v.
        # Therefors, it becomes -R_cw @ t_wc , which is a tvec expected by COLMAP.
        # convert to COLMAP expected (qw, qx, qy, qz)
        # scipy.Rotation.as_quat() returns [x,y,z,w]
        qvec_colmap_xyzw = rot_cam_from_world.as_quat()
        qvec_colmap_wxyz = np.array(
            [
                qvec_colmap_xyzw[3],
                qvec_colmap_xyzw[0],
                qvec_colmap_xyzw[1],
                qvec_colmap_xyzw[2],
            ]
        )

        # process RGB
        image_filename = f"{i:08d}.jpg"
        rgb_source_path = os.path.join(record3d_path, "rgb", image_filename)

        if not os.path.exists(rgb_source_path):
            logger.warning(
                f"cannot find RGB image: {rgb_source_path}. Skipping this frame."
            )
            continue

        # link images to COLMAP's images/ directory.
        target_rgb_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(target_rgb_path):
            try:
                os.link(rgb_source_path, target_rgb_path)
            except OSError:  # in case symbolic link is not supported.
                iio.imwrite(target_rgb_path, iio.imread(rgb_source_path))

        # prepare COLMAP Image object
        cam_from_world = pycolmap.Rigid3d(qvec_colmap_wxyz, t_cam_from_world)

        colmap_images[i + 1] = pycolmap.Image(
            id=i + 1,
            name=image_filename,
            camera_id=1,
            cam_from_world=cam_from_world,
        )

        # create sparse 3D points (for points3D.bin)
        depth_source_path = os.path.join(record3d_path, "depth", f"{i:08d}.exr")
        if not os.path.exists(depth_source_path):
            logger.warning(
                f"cannot find depth exr file: {depth_source_path}. skipping this frame."
            )
            continue

        # Validate EXR file before processing
        if not validate_exr_file(depth_source_path):
            logger.warning(
                f"depth exr file: {depth_source_path} is invalid. Skipping this frame."
            )
            continue

        success, depth_image = read_exr_depth_safe(depth_source_path)
        if not success or depth_image is None:
            logger.error(
                f"failed to load depth exr file: {depth_source_path}. Skipping this frame."
            )
            continue

        # Check if depth data is valid
        if np.all(np.isnan(depth_image)) or np.all(depth_image == 0):
            logger.warning(
                f"depth file: {depth_source_path} has no valid depth data. Skipping this frame."
            )
            continue

        # get intrainsic (in case different for the frame)
        current_intrinsics = metadata["perFrameIntrinsicCoeffs"][i]
        fx, fy, cx, cy = (
            current_intrinsics[0],
            current_intrinsics[1],
            current_intrinsics[2],
            current_intrinsics[3],
        )

        # sample 3D points randomly for COLMAP sparce poiints
        actual_height, actual_width = depth_image.shape[:2]
        num_pixels = actual_height * actual_width
        num_samples_current_frame = min(num_points_to_sample, num_pixels)
        sample_indices = np.random.choice(
            num_pixels, num_samples_current_frame, replace=False
        )
        # convert 1d to 2d indices.
        rows, cols = np.unravel_index(sample_indices, (actual_height, actual_width))

        # Load RGB image once per frame (outside the loop)
        rgb_image_data = iio.imread(rgb_source_path)

        # Scale coordinates if depth and RGB have different resolutions
        scale_y = rgb_height / actual_height
        scale_x = rgb_width / actual_width

        for r, c in zip(rows, cols):
            depth_val = depth_image[r, c]
            # treat only valid depth
            if depth_val > 0 and not np.isinf(depth_val) and not np.isnan(depth_val):
                # Scale coordinates to match RGB image coordinate system
                scaled_c = c * scale_x
                scaled_r = r * scale_y

                # compute 3D point in camera coordinates
                # +X right, +Y down, +Z front)
                x_cam = (scaled_c - cx) * depth_val / fx
                y_cam = (scaled_r - cy) * depth_val / fy
                z_cam = depth_val

                point_cam = np.array([x_cam, y_cam, z_cam])

                # transform to world coordinates
                # P_world = R_wc @ P_cam + t_wc
                point_world = rot_world_from_cam.apply(point_cam) + t_world_from_cam

                # get RGB (scaled coordinates for RGB sampling)
                # Ensure we're within RGB image bounds
                rgb_r = min(int(scaled_r), rgb_height - 1)
                rgb_c = min(int(scaled_c), rgb_width - 1)
                color = rgb_image_data[rgb_r, rgb_c, :3]  # get R, G, B (0-255)

                # add to COLMAP Point3D object
                colmap_points3D[point_id_counter] = {
                    "xyz": point_world,
                    "color": color.astype(np.uint8),
                }

                point_id_counter += 1

        # if (i + 1) % 100 == 0:
        logger.info(
            f"processed {i + 1} frames. current number of 3D points: {len(colmap_points3D)}\r"
        )

    # 3. write to COLMAP .bin
    try:
        # Create reconstruction object and write to files
        reconstruction = pycolmap.Reconstruction()

        # Add cameras with explicit ID verification
        for camera_id, camera in colmap_cameras.items():
            # Ensure camera has the correct ID set
            if not hasattr(camera, "camera_id") or camera.camera_id != camera_id:
                camera.camera_id = camera_id
            reconstruction.add_camera(camera)

        # Add images (IDs are already set in constructor)
        for image_id, image in colmap_images.items():
            reconstruction.add_image(image)

        # Add points3D using the correct method signature
        for point3d_id, point_data in colmap_points3D.items():
            # The add_point3D method expects: xyz, track, color
            track = pycolmap.Track()  # Create empty track
            reconstruction.add_point3D(
                xyz=point_data["xyz"], track=track, color=point_data["color"]
            )

        # Write the reconstruction
        reconstruction.write_binary(sparse_dir)

        logger.info(f"wrote to COLMAP .bin: {sparse_dir}")
        logger.info(
            f"total num cameras: {len(colmap_cameras)}, images: {len(colmap_images)}, 3D points: {len(colmap_points3D)}"
        )
        return True

    except Exception as e:
        logger.error(f"failed to write into COLMAP .bin: {e}")
        logger.error("check pycolmap is properly installed and working.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Record3D data to COLMAP format"
    )
    parser.add_argument(
        "--record3d_path",
        type=str,
        default="data/sample/",
        help="Path to Record3D exported data directory (default: data/sample/)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for COLMAP data (default: same as record3d_path)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000,
        help="Number of points to sample per frame for points3D.bin (default: 1000)",
    )

    args = parser.parse_args()

    record3d_path = args.record3d_path
    colmap_output_path = args.output_path if args.output_path else record3d_path
    num_points_to_sample = args.num_points

    if not os.path.exists(record3d_path):
        logger.error(f"There is no Record3D data path: {record3d_path}")
        logger.error("set proper --record3d_path")
        return
    elif not os.path.exists(colmap_output_path):
        logger.warning(
            f"There is no COLMAP output path: {colmap_output_path}. Creating it..."
        )
        os.makedirs(colmap_output_path, exist_ok=True)

    if convert_record3d_to_colmap(
        record3d_path, colmap_output_path, num_points_to_sample
    ):
        logger.info("\n変換が完了しました。")
        print("EDGSの学習を実行するには、以下のコマンドを参考にしてください:")
        print("CUDA_VISIBLE_DEVICES=0 python train.py \\")
        print("train.gs_epochs=30000 \\")
        print("train.no_densify=True \\")
        print(f"gs.dataset.source_path={colmap_output_path} \\")
        print("gs.dataset.model_path=/path/to/output_model_folder \\")
        print("init_wC.matches_per_ref=20000 \\")
        print("init_wC.nns_per_ref=3 \\")
        print("init_wC.num_refs=180")
    else:
        logger.error("\nFailed to conversion. See the logs for details.")


if __name__ == "__main__":
    main()
