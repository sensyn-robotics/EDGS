import json
import os
import subprocess
import sys

import imageio.v3 as iio  # For reading EXR and PNG
import numpy as np
import pycolmap  # COLMAP .binファイル書き込みに必要
from scipy.spatial.transform import Rotation as R

# --- 設定 ---
# Record3Dデータのエクスポートディレクトリへのパスを設定してください
RECORD3D_DATA_PATH = "data/office_fakegreen/"
# 生成されるCOLMAPデータセットの出力ディレクトリへのパスを設定してください
COLMAP_OUTPUT_PATH = "data/office_fakegreen/"

# points3D.binに含める3D点のサンプリング数 (フレームごとに)
# EDGSは通常、独自の密な初期化を使用しますが、COLMAP形式の要件として、
# スパースな点群を含めることが一般的です。
NUM_POINTS_TO_SAMPLE_FOR_POINTS3D_BIN = 1000


def validate_exr_file(file_path):
    """
    Validate if an EXR file is readable and contains valid depth data.

    Args:
        file_path: Path to the EXR file

    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        # Check file size
        if os.path.getsize(file_path) < 100:  # Very small files are likely corrupted
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
                print(
                    f"EXR read error for {os.path.basename(exr_path)}: {result.stderr.strip()}"
                )
            return False, None

    except subprocess.TimeoutExpired:
        print(f"Timeout reading {os.path.basename(exr_path)}")
        return False, None
    except Exception as e:
        print(f"EXR read subprocess error for {os.path.basename(exr_path)}: {e}")
        return False, None


# --- COLMAP形式データセット変換関数 ---
def convert_record3d_to_colmap(record3d_path, colmap_output_path):
    print(
        f"Record3DデータをCOLMAP形式に変換中: {record3d_path} -> {colmap_output_path}"
    )

    # 1. COLMAPディレクトリ構造の作成
    # COLMAPは `images/` と `sparse/0/` の構造を期待します [2]。
    images_dir = os.path.join(colmap_output_path, "images")
    sparse_dir = os.path.join(colmap_output_path, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # 2. metadata.jsonの読み込み
    metadata_path = os.path.join(record3d_path, "metadata.json")
    if not os.path.exists(metadata_path):
        print(
            f"エラー: metadata.jsonが見つかりません。パスを確認してください: {metadata_path}"
        )
        return False

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # カメラの幅と高さはメタデータから取得 [3]
    width = metadata["w"]
    height = metadata["h"]

    # COLMAP Cameraオブジェクトの準備: ID, MODEL, WIDTH, HEIGHT, PARAMS
    # Record3Dは通常、カメラの内部パラメータが一定であるため、単一のカメラモデルを使用します。
    # metadata.jsonの`perFrameIntrinsicCoeffs`は各フレームの `[fx, fy, cx, cy]` を提供します [4, 5]。
    # ここでは最初のフレームの内部パラメータを代表として使用します。
    first_intrinsics = metadata["perFrameIntrinsicCoeffs"][0]

    # COLMAPのPINHOLEモデルはfx, fy, cx, cyを期待します。
    colmap_cameras = {
        1: pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[
                first_intrinsics[0],
                first_intrinsics[1],
                first_intrinsics[2],
                first_intrinsics[3],
            ],
        )
    }

    colmap_images = {}
    colmap_points3D = {}
    point_id_counter = 1  # 3D点のユニークなID

    print("各フレームの処理を開始...")
    # 各フレームの処理
    for i, frame_pose_data in enumerate(metadata["poses"]):
        # Record3Dの`poses`は `[qx, qy, qz, qw, tx, ty, tz]` の順で提供されます [3, 9-26]。
        # これは、カメラ座標系からワールド座標系への変換 (world_from_camera) を表します。
        # P_world = R_wc @ P_cam + t_wc

        # COLMAPのimages.binは、ワールド座標系からカメラ座標系への変換 (camera_from_world) を期待します。
        # P_cam = R_cw @ P_world + t_cw
        # ここで、R_cw = R_wc.T (転置) および t_cw = -R_cw @ t_wc となります。

        # Record3Dのクォータニオンと並進ベクトルを抽出
        quat_record3d_xyzw = np.array(frame_pose_data[:4])
        t_world_from_cam = np.array(frame_pose_data[4:])

        # Record3Dのクォータニオン (scipy.spatial.transform.Rotationは [x,y,z,w] 順を期待) からRotationオブジェクトを作成
        rot_world_from_cam = R.from_quat(quat_record3d_xyzw)

        # ポーズを逆転させて、world_to_camera (camera_from_world) 変換を取得
        rot_cam_from_world = rot_world_from_cam.inv()
        t_cam_from_world = -rot_cam_from_world.apply(t_world_from_cam)
        # scipy.Rotation.apply(v) は R @ v を計算します。
        # よって、-R_cw @ t_wc となり、これがCOLMAPが期待するtvecです。

        # COLMAPが期待するクォータニオン形式 (qw, qx, qy, qz) に変換
        # scipy.Rotation.as_quat() は [x,y,z,w] 順を返します。
        qvec_colmap_xyzw = rot_cam_from_world.as_quat()
        qvec_colmap_wxyz = np.array(
            [
                qvec_colmap_xyzw[3],
                qvec_colmap_xyzw[0],
                qvec_colmap_xyzw[1],
                qvec_colmap_xyzw[2],
            ]
        )

        # RGB画像ファイルの処理
        image_filename = f"{i:08d}.jpg"  # 例: 00000.png, 00001.png
        rgb_source_path = os.path.join(record3d_path, "rgb", image_filename)

        if not os.path.exists(rgb_source_path):
            print(
                f"警告: RGB画像 {rgb_source_path} が見つかりません。このフレームをスキップします。"
            )
            continue

        # COLMAPのimagesディレクトリに画像をコピー (またはシンボリックリンク)
        # 大規模データセットではシンボリックリンクが効率的です。
        target_rgb_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(target_rgb_path):
            try:
                os.link(rgb_source_path, target_rgb_path)
            except (
                OSError
            ):  # シンボリックリンクがサポートされていないファイルシステムの場合
                iio.imwrite(
                    target_rgb_path, iio.imread(rgb_source_path)
                )  # 画像を読み込んで書き込む

        # COLMAP Imageオブジェクトの準備
        cam_from_world = pycolmap.Rigid3d(qvec_colmap_wxyz, t_cam_from_world)

        colmap_images[i + 1] = pycolmap.Image(
            name=image_filename,
            keypoints=np.array([]).reshape(0, 2).astype(np.float64),
            cam_from_world=cam_from_world,
            camera_id=1,
            id=i + 1,
        )

        # 深度画像からのスパース3D点群の生成 (points3D.bin用)
        depth_source_path = os.path.join(record3d_path, "depth", f"{i:08d}.exr")
        if not os.path.exists(depth_source_path):
            print(
                f"警告: 深度画像 {depth_source_path} が見つかりません。このフレームの3D点生成をスキップします。"
            )
            continue

        # Validate EXR file before processing
        if not validate_exr_file(depth_source_path):
            print(
                f"警告: 深度画像 {depth_source_path} は破損しているか無効です。このフレームの3D点生成をスキップします。"
            )
            continue

        success, depth_image = read_exr_depth_safe(depth_source_path)
        if not success or depth_image is None:
            print(
                f"エラー: 深度EXR {depth_source_path} の読み込みに失敗しました。このフレームの3D点生成をスキップします。"
            )
            continue

        # Check if depth data is valid
        if np.all(np.isnan(depth_image)) or np.all(depth_image == 0):
            print(
                f"警告: 深度画像 {depth_source_path} に有効な深度データがありません。スキップします。"
            )
            continue

        # このフレームの内部パラメータを取得 (各フレームで異なる場合があるため)
        current_intrinsics = metadata["perFrameIntrinsicCoeffs"][i]
        fx, fy, cx, cy = (
            current_intrinsics[0],
            current_intrinsics[1],
            current_intrinsics[2],
            current_intrinsics[3],
        )

        # 3D点群をランダムにサンプリング
        actual_height, actual_width = depth_image.shape[:2]
        print(
            f"Debug: RGB dimensions: {height}x{width}, Depth dimensions: {actual_height}x{actual_width}"
        )

        # Use actual depth image dimensions for sampling
        num_pixels = actual_height * actual_width
        num_samples_current_frame = min(
            NUM_POINTS_TO_SAMPLE_FOR_POINTS3D_BIN, num_pixels
        )
        sample_indices = np.random.choice(
            num_pixels, num_samples_current_frame, replace=False
        )
        rows, cols = np.unravel_index(sample_indices, (actual_height, actual_width))

        # Load RGB image once per frame (outside the loop)
        rgb_image_data = iio.imread(rgb_source_path)

        # Scale coordinates if depth and RGB have different resolutions
        scale_y = height / actual_height
        scale_x = width / actual_width

        for r, c in zip(rows, cols):
            depth_val = depth_image[r, c]
            # 有効な深度値のみを処理 (0、NaN、無限大は無視)
            if depth_val > 0 and not np.isinf(depth_val) and not np.isnan(depth_val):
                # Scale coordinates to match RGB image coordinate system
                scaled_c = c * scale_x
                scaled_r = r * scale_y

                # ピクセルと深度からカメラ座標での3D点を計算
                # (標準的なコンピュータビジョンカメラフレーム: +X右, +Y下, +Z前)
                x_cam = (scaled_c - cx) * depth_val / fx
                y_cam = (scaled_r - cy) * depth_val / fy
                z_cam = depth_val

                point_cam = np.array([x_cam, y_cam, z_cam])

                # カメラ座標系の点をワールド座標系に変換 (world_from_camera変換を使用)
                # P_world = R_wc @ P_cam + t_wc
                point_world = rot_world_from_cam.apply(point_cam) + t_world_from_cam

                # RGBカラーを取得 (scaled coordinates for RGB sampling)
                # Ensure we're within RGB image bounds
                rgb_r = min(int(scaled_r), height - 1)
                rgb_c = min(int(scaled_c), width - 1)
                color = rgb_image_data[rgb_r, rgb_c, :3]  # R, G, B成分を取得 (0-255)

                # COLMAP Point3Dオブジェクトに追加
                point3d = pycolmap.Point3D()
                point3d.xyz = point_world
                point3d.color = color.astype(np.uint8)
                point3d.error = 0.0
                colmap_points3D[point_id_counter] = point3d

                point_id_counter += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1} フレーム処理済み。現在の3D点数: {len(colmap_points3D)}")

    # 3. COLMAP .binファイルの書き込み
    # pycolmapライブラリの書き込み関数を使用します。
    try:
        # Create reconstruction object and write to files
        reconstruction = pycolmap.Reconstruction()

        # Add cameras with explicit IDs
        for camera_id, camera in colmap_cameras.items():
            camera.camera_id = camera_id  # Set the camera ID explicitly
            reconstruction.add_camera(camera)

        # Add images with explicit IDs
        for image_id, image in colmap_images.items():
            image.image_id = image_id  # Set the image ID explicitly
            reconstruction.add_image(image)

        # Add points3D with explicit IDs
        for point3d_id, point3d in colmap_points3D.items():
            point3d.point3D_id = point3d_id  # Set the point3D ID explicitly
            reconstruction.add_point3d(point3d)

        # Write the reconstruction
        reconstruction.write_binary(sparse_dir)

        print(f"COLMAP .binファイルが正常に書き込まれました: {sparse_dir}")
        print(
            f"合計カメラ数: {len(colmap_cameras)}, 画像数: {len(colmap_images)}, 3D点数: {len(colmap_points3D)}"
        )
        return True

    except Exception as e:
        print(f"エラー: COLMAP .binファイルの書き込み中にエラーが発生しました: {e}")
        print("pycolmapが正しくインストールされ、動作しているか確認してください。")
        return False


# --- スクリプト実行例 ---
if __name__ == "__main__":
    # ここにRecord3DデータとCOLMAP出力のパスを設定してください
    # 例:
    # RECORD3D_DATA_PATH = "/Users/youruser/Documents/Record3D_Exports/2024-01-01_12-00-00"
    # COLMAP_OUTPUT_PATH = "/Users/youruser/Documents/COLMAP_Datasets/my_record3d_scene"

    # 上記のRECORD3D_DATA_PATHとCOLMAP_OUTPUT_PATHを実際のパスに置き換えてください。
    # この例はデモンストレーション用です。

    if not os.path.exists(RECORD3D_DATA_PATH):
        print(
            f"エラー: 指定されたRecord3Dデータパスが存在しません: {RECORD3D_DATA_PATH}"
        )
        print("RECORD3D_DATA_PATH変数を正しいパスに設定してください。")
    elif not os.path.exists(COLMAP_OUTPUT_PATH):
        print(
            f"警告: COLMAP出力パス {COLMAP_OUTPUT_PATH} が存在しません。作成を試みます。"
        )
        os.makedirs(COLMAP_OUTPUT_PATH, exist_ok=True)  # COLMAP_OUTPUT_PATH自体も作成

    if convert_record3d_to_colmap(RECORD3D_DATA_PATH, COLMAP_OUTPUT_PATH):
        print("\n変換が完了しました。")
        print("EDGSの学習を実行するには、以下のコマンドを参考にしてください:")
        print("CUDA_VISIBLE_DEVICES=0 python train.py \\")
        print("train.gs_epochs=30000 \\")
        print("train.no_densify=True \\")
        print(f"gs.dataset.source_path={COLMAP_OUTPUT_PATH} \\")
        print("gs.dataset.model_path=/path/to/output_model_folder \\")
        print("init_wC.matches_per_ref=20000 \\")
        print("init_wC.nns_per_ref=3 \\")
        print("init_wC.num_refs=180")
    else:
        print("\n変換に失敗しました。上記のエラーメッセージを確認してください。")
