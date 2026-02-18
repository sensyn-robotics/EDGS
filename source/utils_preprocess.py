# This file contains function for video or image collection preprocessing.
# For video we do the preprocessing and select k sharpest frames.
# Afterwards scene is constructed
import os
import time
import yaml

import cv2
import numpy as np
import pycolmap
from matplotlib import pyplot as plt
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm


def load_colmap_config(config_name="colmap_03_optimal_quality"):
    """
    Load COLMAP configuration from YAML file.
    
    Args:
        config_name (str): Configuration profile name. Options:
            - 'colmap_01_highest_quality': Best quality, highest memory usage
            - 'colmap_02_high_quality': High quality, good memory efficiency  
            - 'colmap_03_optimal_quality': Balanced quality and memory (default)
            - 'colmap_04_medium_quality': Good quality, memory efficient
            - 'colmap_05_low_quality': Reduced quality, low memory usage
            - 'colmap_06_lowest_quality': Minimal quality, very low memory usage
    
    Returns:
        dict: Configuration dictionary with SIFT and mapping parameters
    """
    # Get the project root directory (assuming this file is in source/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle both old naming (without colmap_ prefix) and new naming
    if not config_name.startswith("colmap_"):
        config_name = f"colmap_{config_name}"
    
    config_file = os.path.join(project_root, "configs", f"{config_name}.yaml")
    
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found, using optimal quality profile")
        config_file = os.path.join(project_root, "configs", "colmap_03_optimal_quality.yaml")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded COLMAP config: {config_name} from {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config {config_file}: {e}")
        print("Using fallback default configuration")
        return get_default_colmap_config()


def get_default_colmap_config():
    """Fallback configuration if YAML files are not available"""
    return {
        'sift_extraction': {
            'max_num_features': 4096,
            'max_image_size': 1920,
            'first_octave': 0,
            'num_octaves': 3,
            'octave_resolution': 3,
            'peak_threshold': 0.008,
            'edge_threshold': 15,
            'estimate_affine_shape': False,
            'domain_size_pooling': False,
            'upright': False,
        },
        'sift_matching': {
            'max_ratio': 0.8,
            'max_distance': 0.75,
            'cross_check': True,
            'max_num_matches': 8192,
        },
        'pipeline': {
            'min_num_matches': 5,
            'multiple_models': False,
            'max_num_models': 1,
            'max_model_overlap': 100,
            'min_model_size': 30,
            'extract_colors': True,
            'num_threads': 4,
        },
        'mapper': {
            'init_min_num_inliers': 12,
            'init_max_error': 8.0,
            'init_min_tri_angle': 1.5,
            'abs_pose_min_num_inliers': 12,
            'abs_pose_max_error': 12.0,
            'abs_pose_min_inlier_ratio': 0.15,
            'filter_max_reproj_error': 8.0,
            'filter_min_tri_angle': 0.25,
        }
    }


def get_rotation_moviepy(video_path):
    clip = VideoFileClip(video_path)
    rotation = 0

    try:
        displaymatrix = clip.reader.infos["inputs"][0]["streams"][2]["metadata"].get(
            "displaymatrix", ""
        )
        if "rotation of" in displaymatrix:
            angle = float(
                displaymatrix.strip().split("rotation of")[-1].split("degrees")[0]
            )
            rotation = int(angle) % 360

    except Exception as e:
        print(f"No displaymatrix rotation found: {e}")

    clip.reader.close()
    # if clip.audio:
    #    clip.audio.reader.close_proc()

    return rotation


def resize_max_side(frame, max_size):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def extract_video_frames_to_disk(video_input, output_dir, k=1, max_size=1024, use_all_frames=False, target_fps=3.0):
    """
    Extracts every k-th frame from a video using ffmpeg, saves to disk to avoid memory overflow.

    Parameters:
        video_input (str, file-like, or list): Path to video file, file-like object, or list of image files.
        output_dir (str): Directory to save extracted frames.
        k (int): Interval for frame extraction (every k-th frame).
        max_size (int): Maximum dimension (width or height) for resizing. Images are resized 
                       to fit within max_size x max_size while preserving aspect ratio.
                       For example: 3840x2160 with max_size=1920 becomes 1920x1080.
        use_all_frames (bool): If True, extract at target_fps. If False, extract every k-th frame.
        target_fps (float): Target frames per second for extraction when use_all_frames=True.

    Returns:
        frame_paths (list): List of paths to extracted frame files.
    """
    import subprocess
    import tempfile
    import shutil
    
    # Handle list of image files (not single video in a list)
    if isinstance(video_input, list):
        # If it's a single video in a list, treat it as video
        if len(video_input) == 1 and video_input[0].name.endswith(
            (".mp4", ".avi", ".mov")
        ):
            video_input = video_input[0]  # unwrap single video file
        else:
            # Treat as list of images - copy and resize them
            frame_paths = []
            for idx, img_file in enumerate(video_input):
                img = Image.open(img_file.name).convert("RGB")
                # Resize if necessary
                width, height = img.size
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                output_path = os.path.join(output_dir, f"frame_{idx:08d}.jpg")
                img.save(output_path, "JPEG", quality=95)
                frame_paths.append(output_path)
            return frame_paths

    # Handle file-like or path
    if hasattr(video_input, "name"):
        video_path = video_input.name
    elif isinstance(video_input, (str, os.PathLike)):
        video_path = str(video_input)
    else:
        raise ValueError(
            "Unsupported video input type. Must be a filepath, file-like object, or list of images."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ffmpeg to extract frames
    print(f"Extracting frames from video using ffmpeg...")
    try:
        # First, get video info to calculate frame interval
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True, check=True)
        
        frame_count_str = result.stdout.strip()
        if frame_count_str and frame_count_str != 'N/A':
            total_frames = int(frame_count_str)
            print(f"Total frames in video: {total_frames}")
        else:
            print("Could not determine frame count, using default extraction")
            total_frames = 1000  # Default fallback
        
        # Extract frames at target fps using ffmpeg with scaling
        scale_filter = f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease"
        fps_filter = f"fps={target_fps}"
        
        # Calculate expected number of frames based on video duration and target fps
        # This ensures we don't artificially limit frame extraction
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_path, '-y',
            '-vf', f'{fps_filter},{scale_filter}',
            '-q:v', '2',  # High quality
            os.path.join(output_dir, 'frame_%08d.jpg')
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Get list of extracted frame paths
        frame_paths = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.startswith('frame_') and f.endswith('.jpg')
        ])
        
        print(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e}")
        # Fallback to opencv if ffmpeg fails
        return extract_video_frames_fallback(video_path, output_dir, k, max_size, target_fps)
    except FileNotFoundError:
        print("ffmpeg not found, using opencv fallback")
        return extract_video_frames_fallback(video_path, output_dir, k, max_size, target_fps)


def extract_video_frames_fallback(video_path, output_dir, k=1, max_size=1024, target_fps=3.0):
    """
    Fallback method using opencv, but saves frames to disk instead of memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    frame_paths = []
    
    print(f"Video info: {width}x{height}, {video_fps:.1f} fps, {total_frames} frames reported")
    
    # Handle videos with incorrect frame count or very high resolution
    if total_frames > 1000000:  # If more than 1M frames, likely incorrect
        print(f"Warning: Video reports {total_frames} frames, which seems incorrect. Limiting extraction.")
        # Try to extract frames until we can't read anymore
        max_frames_to_extract = 100000  # Try up to 100k frame reads
    else:
        max_frames_to_extract = min(total_frames, 5000)  # Limit to 5k frames max
        
    # Additional settings for high resolution videos
    if width * height > 1920 * 1080:  # If higher than 1080p
        print(f"High resolution video detected ({width}x{height})")
    
    # Calculate frame interval based on target fps
    # Handle incorrect FPS detection (common with some video codecs)
    if video_fps > 1000:  # Clearly incorrect FPS
        print(f"Warning: Detected FPS ({video_fps}) seems incorrect. Using frame count to estimate.")
        # Estimate actual FPS assuming ~30-60 fps is typical
        estimated_duration = total_frames / 30.0  # Assume 30fps as baseline
        frame_interval = max(1, int(total_frames / (estimated_duration * target_fps)))
        if frame_interval > 100:  # Still too high, use a reasonable default
            frame_interval = max(1, int(30 / target_fps))  # Assume 30fps
            print(f"Using default frame interval: every {frame_interval} frames")
        else:
            print(f"Estimated frame interval: every {frame_interval} frames")
    elif video_fps > 0:
        frame_interval = max(1, int(video_fps / target_fps))
        print(f"Video FPS: {video_fps:.1f}, Target FPS: {target_fps}, Frame interval: every {frame_interval} frames")
    else:
        frame_interval = k  # Fallback to provided k value

    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=min(total_frames, max_frames_to_extract) // frame_interval, desc="Extracting Video Frames", unit="frame") as pbar:
        consecutive_failures = 0
        while frame_count < max_frames_to_extract and consecutive_failures < 10:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"Too many consecutive frame read failures, stopping extraction")
                    break
                # Skip some frames to try to get past corrupted sections
                if consecutive_failures > 3:
                    for _ in range(10):  # Skip 10 frames
                        cap.read()
                    frame_count += 10
                continue
            else:
                consecutive_failures = 0  # Reset on successful read
            if frame_count % frame_interval == 0:
                # Resize frame
                h, w = frame.shape[:2]
                scale = max(h, w) / max_size
                if scale > 1:
                    frame = cv2.resize(frame, (int(w / scale), int(h / scale)))
                
                # Save frame to disk
                frame_path = os.path.join(output_dir, f"frame_{len(frame_paths):08d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(frame_path)
                pbar.update(1)
            frame_count += 1

    cap.release()
    return frame_paths


def resize_max_side(frame, max_size):
    """
    Resizes the frame so that its largest side equals max_size, maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    max_dim = max(height, width)

    if max_dim <= max_size:
        return frame  # No need to resize

    scale = max_size / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_frame


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def process_all_frames(
    IMG_FOLDER="/scratch/datasets/hq_data/night2_all_frames",
    to_visualize=False,
    save_images=True,
):
    dict_scores = {}
    for idx, img_name in tqdm(
        enumerate(sorted([x for x in os.listdir(IMG_FOLDER) if ".png" in x]))
    ):
        img = cv2.imread(os.path.join(IMG_FOLDER, img_name))  # [250:, 100:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )
        if to_visualize:
            plt.figure()
            plt.title(f"Laplacian score: {fm:.2f}")
            plt.imshow(img[..., [2, 1, 0]])
            plt.show()
        dict_scores[idx] = {"idx": idx, "img_name": img_name, "score": fm}
        if save_images:
            dict_scores[idx]["img"] = img

    return dict_scores


def select_optimal_frames(scores, k):
    """
    Selects a minimal subset of frames while ensuring no gaps exceed k.

    Args:
        scores (list of float): List of scores where index represents frame number.
        k (int): Maximum allowed gap between selected frames.

    Returns:
        list of int: Indices of selected frames.
    """
    n = len(scores)
    selected = [0, n - 1]
    i = 0  # Start at the first frame

    while i < n:
        # Find the best frame to select within the next k frames
        best_idx = max(
            range(i, min(i + k + 1, n)), key=lambda x: scores[x], default=None
        )

        if best_idx is None:
            break  # No more frames left

        selected.append(best_idx)
        i = best_idx + k + 1  # Move forward, ensuring gaps stay within k

    return sorted(selected)


def variance_of_laplacian(image):
    """
    Compute the variance of Laplacian as a focus measure.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def preprocess_frame_paths(frame_paths, verbose=False):
    """
    Compute sharpness scores for a list of frame files using multi-scale Laplacian variance.

    Args:
        frame_paths (list of str): List of paths to frame image files.
        verbose (bool): If True, print scores.

    Returns:
        list of float: Sharpness scores for each frame.
    """
    scores = []

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Scoring frames")):
        # Load frame from disk
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load frame {frame_path}")
            scores.append(0.0)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )

        if verbose:
            print(f"Frame {idx} ({os.path.basename(frame_path)}): Sharpness Score = {fm:.2f}")

        scores.append(fm)

    return scores


def preprocess_frames(frames, verbose=False):
    """
    Compute sharpness scores for a list of frames using multi-scale Laplacian variance.
    DEPRECATED: Use preprocess_frame_paths instead to avoid memory issues.

    Args:
        frames (list of np.ndarray): List of frames (BGR images).
        verbose (bool): If True, print scores.

    Returns:
        list of float: Sharpness scores for each frame.
    """
    scores = []

    for idx, frame in enumerate(tqdm(frames, desc="Scoring frames")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )

        if verbose:
            print(f"Frame {idx}: Sharpness Score = {fm:.2f}")

        scores.append(fm)

    return scores


def select_optimal_frames(scores, k):
    """
    Selects k frames by splitting into k segments and picking the sharpest frame from each.

    Args:
        scores (list of float): List of sharpness scores.
        k (int): Number of frames to select.

    Returns:
        list of int: Indices of selected frames.
    """
    n = len(scores)
    selected_indices = []
    segment_size = n // k

    for i in range(k):
        start = i * segment_size
        end = (i + 1) * segment_size if i < k - 1 else n  # Last chunk may be larger
        segment_scores = scores[start:end]

        if len(segment_scores) == 0:
            continue  # Safety check if some segment is empty

        best_in_segment = start + np.argmax(segment_scores)
        selected_indices.append(best_in_segment)

    return sorted(selected_indices)


def copy_selected_frames_to_scene_dir(selected_frame_paths, scene_dir):
    """
    Copies selected frame files into the target scene directory under 'images/' subfolder.

    Args:
        selected_frame_paths (list of str): List of paths to selected frame files.
        scene_dir (str): Target path where 'images/' subfolder will be created.
    """
    import shutil
    
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, frame_path in enumerate(selected_frame_paths):
        filename = os.path.join(
            images_dir, f"{idx:08d}.jpg"
        )  # 00000000.jpg, 00000001.jpg, etc.
        shutil.copy2(frame_path, filename)

    print(f"Copied {len(selected_frame_paths)} selected frames to {images_dir}")


def save_frames_to_scene_dir(frames, scene_dir):
    """
    Saves a list of frames into the target scene directory under 'images/' subfolder.
    DEPRECATED: Use copy_selected_frames_to_scene_dir to avoid memory issues.

    Args:
        frames (list of np.ndarray): List of frames (BGR images) to save.
        scene_dir (str): Target path where 'images/' subfolder will be created.
    """
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        filename = os.path.join(
            images_dir, f"{idx:08d}.png"
        )  # 00000000.png, 00000001.png, etc.
        cv2.imwrite(filename, frame)

    print(f"Saved {len(frames)} frames to {images_dir}")


def create_fallback_reconstruction(image_dir, sparse_path):
    """
    Create a minimal fallback reconstruction when COLMAP fails completely.
    Creates a simple linear camera trajectory for the available images.
    """
    # No need to import colmap_loader - we'll create text files directly
    
    print("🔧 Creating fallback reconstruction with assumed camera positions...")
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) < 1:
        raise RuntimeError("No images found for fallback reconstruction")
    
    # Read first image to get dimensions
    first_img_path = os.path.join(image_dir, image_files[0])
    from PIL import Image
    img = Image.open(first_img_path)
    width, height = img.size
    
    # Create minimal reconstruction directory
    fallback_dir = os.path.join(sparse_path, "0")
    os.makedirs(fallback_dir, exist_ok=True)
    
    # Create cameras.txt with simple pinhole model
    cameras_txt = os.path.join(fallback_dir, "cameras.txt")
    focal = max(width, height)  # Simple focal length estimation
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")
    
    # Create images.txt with linear trajectory
    images_txt = os.path.join(fallback_dir, "images.txt")
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, img_file in enumerate(image_files):
            # Simple linear trajectory along Z-axis
            tx, ty, tz = 0.0, 0.0, -i * 0.5
            # Identity quaternion (no rotation)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_file}\n")
            f.write("\n")  # Empty line for points2D
    
    # Create minimal points3D.txt with a few dummy points
    points_txt = os.path.join(fallback_dir, "points3D.txt")
    with open(points_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Add some dummy 3D points for basic initialization
        for i in range(10):
            x, y, z = i * 0.1, 0.0, -1.0  # Simple grid of points
            r, g, b = 128, 128, 128  # Gray color
            error = 1.0
            f.write(f"{i+1} {x} {y} {z} {r} {g} {b} {error}\n")
    
    print(f"✅ Created fallback reconstruction with {len(image_files)} cameras at {fallback_dir}")
    print("⚠️  Note: This is a basic reconstruction with assumed camera positions. Results may be limited.")


def run_colmap_on_scene(scene_dir, force_pinhole=True, colmap_config="colmap_03_optimal_quality", single_camera=False):
    """
    Runs feature extraction, matching, and mapping on all images inside scene_dir/images using pycolmap.
    Forces PINHOLE camera model to avoid distortion issues.

    Args:
        scene_dir (str): Path to scene directory containing 'images' folder.
        force_pinhole (bool): If True, forces PINHOLE camera model during reconstruction.
        colmap_config: Either a string config name or a dictionary with config settings.
            If string, options are: 'high_accuracy', 'balanced', 'low_memory', 'very_low_memory'
            If dict, should contain the full configuration.
        single_camera (bool): If True, forces all images to share identical camera intrinsics.
            Useful for 360 video processing where all perspective views have the same FOV.
    """
    start_time = time.time()
    print(f"Running COLMAP pipeline on all images inside {scene_dir}")
    
    # Load configuration - handle both string names and dict configs
    if isinstance(colmap_config, str):
        print(f"Using COLMAP configuration profile: {colmap_config}")
        config = load_colmap_config(colmap_config)
    else:
        # Already a config dictionary
        config_name = colmap_config.get('config_name', 'custom')
        print(f"Using COLMAP configuration: {config_name}")
        config = colmap_config

    # Setup paths
    database_path = os.path.join(scene_dir, "database.db")
    sparse_path = os.path.join(scene_dir, "sparse")
    image_dir = os.path.join(scene_dir, "images")

    # Make sure output directories exist
    os.makedirs(sparse_path, exist_ok=True)

    # Step 1: Feature Extraction using configuration
    sift_options = config['sift_extraction']

    # Configure extraction parameters for single camera mode (360 video processing)
    if single_camera:
        print("📷 Single camera mode enabled - all images will share identical intrinsics")
        pycolmap.extract_features(
            database_path,
            image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model='PINHOLE',
            sift_options=sift_options,
        )
    else:
        pycolmap.extract_features(
            database_path,
            image_dir,
            sift_options=sift_options,
        )
    print(f"Finished feature extraction in {(time.time() - start_time):.2f}s.")

    # Step 2: Feature Matching using configuration
    matching_config = config['sift_matching']
    matching_options = pycolmap.SiftMatchingOptions()
    matching_options.max_ratio = matching_config['max_ratio']
    matching_options.max_distance = matching_config['max_distance']
    matching_options.cross_check = matching_config['cross_check']
    if 'max_num_matches' in matching_config:
        matching_options.max_num_matches = matching_config['max_num_matches']
    
    # Use COLMAP automatic_reconstructor matching strategy
    # Match based on data type and dataset size (similar to COLMAP's logic)
    # Reference: colmap/src/colmap/controllers/automatic_reconstruction.cc:209-232

    # Count number of images in database
    database = pycolmap.Database(database_path)
    num_images = database.num_images
    database.close()

    print(f"Database contains {num_images} images")

    # Determine matching strategy based on image count
    # COLMAP uses: < 200 images OR no vocab tree -> exhaustive
    #              >= 200 images AND vocab tree -> vocab tree matching
    # For video data, COLMAP uses sequential matching

    # Check if vocab tree is available
    vocab_tree_path = config.get('vocab_tree_path', None)
    data_type = config.get('data_type', 'individual')  # 'video', 'individual', or 'internet'

    try:
        if data_type == 'video':
            # For video data: use sequential matching
            print("Using sequential matching for video data...")
            pycolmap.match_sequential(database_path,
                                    sift_options=matching_options,
                                    overlap=20,
                                    quadratic_overlap=False)
            print("Sequential matching completed.")
        elif vocab_tree_path and num_images >= 200:
            # For large datasets with vocab tree: use vocabulary tree matching
            print(f"Using vocabulary tree matching for {num_images} images...")
            print(f"Vocab tree path: {vocab_tree_path}")
            pycolmap.match_vocabtree(database_path,
                                   vocab_tree_path=vocab_tree_path,
                                   sift_options=matching_options)
            print("Vocabulary tree matching completed.")
        else:
            # For small datasets or without vocab tree: use exhaustive matching
            print(f"Using exhaustive matching for {num_images} images...")
            pycolmap.match_exhaustive(database_path, sift_options=matching_options)
            print("Exhaustive matching completed.")
    except Exception as e:
        print(f"Matching failed: {e}")
        print("Falling back to exhaustive matching...")
        pycolmap.match_exhaustive(database_path, sift_options=matching_options)
    
    print(f"Finished feature matching in {(time.time() - start_time):.2f}s.")

    # Step 3: Mapping using configuration
    pipeline_config = config['pipeline']
    mapper_config = config['mapper']
    
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    
    # Set pipeline options from configuration
    pipeline_options.min_num_matches = pipeline_config['min_num_matches']
    pipeline_options.multiple_models = pipeline_config['multiple_models']
    pipeline_options.max_num_models = pipeline_config['max_num_models']
    pipeline_options.max_model_overlap = pipeline_config['max_model_overlap']
    pipeline_options.min_model_size = pipeline_config['min_model_size']
    pipeline_options.extract_colors = pipeline_config['extract_colors']
    pipeline_options.num_threads = pipeline_config['num_threads']
    
    # Set mapper options from configuration
    pipeline_options.mapper.init_min_num_inliers = mapper_config['init_min_num_inliers']
    pipeline_options.mapper.init_max_error = mapper_config['init_max_error']
    pipeline_options.mapper.init_min_tri_angle = mapper_config['init_min_tri_angle']
    pipeline_options.mapper.abs_pose_min_num_inliers = mapper_config['abs_pose_min_num_inliers']
    pipeline_options.mapper.abs_pose_max_error = mapper_config['abs_pose_max_error']
    pipeline_options.mapper.abs_pose_min_inlier_ratio = mapper_config['abs_pose_min_inlier_ratio']
    pipeline_options.mapper.filter_max_reproj_error = mapper_config['filter_max_reproj_error']
    pipeline_options.mapper.filter_min_tri_angle = mapper_config['filter_min_tri_angle']
    
    # Bundle adjustment refinement settings (try to set if available)
    try:
        # When single_camera mode is enabled, disable focal length refinement
        # to keep all cameras with identical intrinsics (important for 360 processing)
        if single_camera:
            pipeline_options.mapper.global_ba_refine_focal_length = False
            pipeline_options.mapper.global_ba_refine_principal_point = False
            pipeline_options.mapper.global_ba_refine_extra_params = False
            print("🔒 Disabled focal length refinement for single camera mode")
        else:
            pipeline_options.mapper.global_ba_refine_focal_length = True
            pipeline_options.mapper.global_ba_refine_principal_point = False
            pipeline_options.mapper.global_ba_refine_extra_params = False
    except AttributeError:
        pass  # Skip if not available in this pycolmap version
    
    # Note: force_pinhole will be applied after reconstruction

    try:
        reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_path,
            options=pipeline_options,
        )
        print(f"Finished incremental mapping in {(time.time() - start_time):.2f}s.")
    except Exception as e:
        print(f"⚠️  Initial reconstruction failed: {e}")
        print("🔄 Trying with even more lenient settings...")
        
        # Try with ultra-lenient settings as fallback
        pipeline_options.min_num_matches = 5
        pipeline_options.min_model_size = 2
        pipeline_options.mapper.init_min_num_inliers = 10
        pipeline_options.mapper.init_max_error = 20.0
        pipeline_options.mapper.init_min_tri_angle = 1.0
        
        try:
            reconstruction = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=image_dir,
                output_path=sparse_path,
                options=pipeline_options,
            )
            print(f"✅ Fallback reconstruction succeeded in {(time.time() - start_time):.2f}s.")
        except Exception as e2:
            print(f"❌ Both reconstruction attempts failed: {e2}")
            raise RuntimeError("COLMAP reconstruction failed. The video might have insufficient overlap or features.")

    # Step 4: Merge multiple reconstructions if they exist
    target_recon_path = os.path.join(sparse_path, "0")
    
    # Find all reconstructions
    reconstructions_found = []
    for i in range(100):  # Check more reconstruction indices
        alt_path = os.path.join(sparse_path, str(i))
        if os.path.exists(alt_path) and any(os.path.exists(os.path.join(alt_path, f)) 
                                         for f in ["cameras.bin", "images.bin", "points3D.bin"]):
            reconstructions_found.append((i, alt_path))
    
    if not reconstructions_found:
        print("❌ COLMAP reconstruction failed - creating minimal fallback reconstruction")
        return create_fallback_reconstruction(image_dir, sparse_path)
    
    print(f"🔍 Found {len(reconstructions_found)} reconstructions: {[i for i, _ in reconstructions_found]}")
    
    if len(reconstructions_found) == 1:
        # Single reconstruction - just ensure it's at index 0
        idx, recon_path = reconstructions_found[0]
        if idx != 0:
            import shutil
            if os.path.exists(target_recon_path):
                shutil.rmtree(target_recon_path)
            shutil.move(recon_path, target_recon_path)
            print(f"📁 Moved single reconstruction from {idx} to 0")
        recon_path = target_recon_path
    else:
        # Multiple reconstructions - try to merge them
        print("🔄 Attempting to merge multiple reconstructions...")
        try:
            merged_reconstruction = merge_colmap_reconstructions(reconstructions_found, sparse_path)
            merged_reconstruction.write(target_recon_path)
            print(f"✅ Successfully merged {len(reconstructions_found)} reconstructions")
            recon_path = target_recon_path
        except Exception as e:
            print(f"⚠️  Failed to merge reconstructions: {e}")
            # Fall back to using the largest reconstruction
            largest_idx, largest_path = max(reconstructions_found, 
                                          key=lambda x: get_reconstruction_size(x[1]))
            print(f"📊 Using largest reconstruction: {largest_idx}")
            if largest_idx != 0:
                import shutil
                if os.path.exists(target_recon_path):
                    shutil.rmtree(target_recon_path)
                shutil.move(largest_path, target_recon_path)
            recon_path = target_recon_path

    # Step 5: Convert cameras to PINHOLE if needed
    if os.path.exists(recon_path):
        reconstruction = pycolmap.Reconstruction(recon_path)
        
        if len(reconstruction.cameras) == 0:
            raise RuntimeError("❌ Reconstruction contains no cameras")
        if len(reconstruction.images) == 0:
            raise RuntimeError("❌ Reconstruction contains no images")
        if len(reconstruction.points3D) == 0:
            print("⚠️  Warning: Reconstruction contains no 3D points")

        for cam in reconstruction.cameras.values():
            if force_pinhole and cam.model != "PINHOLE":
                print(f"Converting camera {cam.camera_id} from {cam.model} to PINHOLE")
                cam.model = "PINHOLE"
                # Ensure we have exactly 4 parameters [fx, fy, cx, cy]
                if len(cam.params) >= 4:
                    cam.params = cam.params[:4]
                elif len(cam.params) >= 3:
                    # Duplicate focal length if we only have 3 params
                    f, cx, cy = cam.params[:3]
                    cam.params = [f, f, cx, cy]
                else:
                    # Default values if params are insufficient
                    focal = max(cam.width, cam.height)
                    cam.params = [focal, focal, cam.width/2, cam.height/2]

        reconstruction.write(recon_path)
        print(f"✅ Saved reconstruction with PINHOLE cameras to {recon_path}")
        print(f"📊 Reconstruction stats: {len(reconstruction.cameras)} cameras, {len(reconstruction.images)} images, {len(reconstruction.points3D)} points")

    print(f"Total pipeline time: {(time.time() - start_time):.2f}s.")


def process_input_for_colmap(input_path, num_ref_views, output_dir, max_size=1024, use_all_frames=False, target_fps=3.0):
    """
    Memory-efficient helper function to process video/images, select optimal frames,
    and save them to the output_dir/images without loading all frames into memory.
    
    Args:
        input_path: Path to video or image directory
        num_ref_views: Number of reference views to select (ignored if use_all_frames=True)
        output_dir: Output directory for processed images
        max_size: Maximum dimension (width or height) for image resizing. Images are resized 
                 to fit within max_size x max_size while preserving aspect ratio.
                 Example: 4K (3840x2160) with max_size=1920 becomes 1920x1080.
        use_all_frames: If True, use all frames (or sample densely) instead of selecting optimal ones
        target_fps: Target frames per second for video extraction when use_all_frames=True
    """
    import tempfile
    import shutil
    
    # Create temporary directory for extracted frames
    temp_frames_dir = tempfile.mkdtemp(prefix="edgs_frames_")
    
    try:
        if isinstance(input_path, (str, os.PathLike)):  # If input_path is a path string
            if os.path.isdir(input_path):  # If it's a directory of images
                print(f"Processing image directory: {input_path}")
                # Copy and resize images to temp directory
                frame_paths = []
                image_files = sorted([
                    f for f in os.listdir(input_path)
                    if f.lower().endswith(("jpg", "jpeg", "png"))
                ])
                
                for idx, img_file in enumerate(image_files):
                    img = Image.open(os.path.join(input_path, img_file)).convert("RGB")
                    # Resize if necessary
                    width, height = img.size
                    if max(width, height) > max_size:
                        scale = max_size / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    output_path = os.path.join(temp_frames_dir, f"frame_{idx:08d}.jpg")
                    img.save(output_path, "JPEG", quality=95)
                    frame_paths.append(output_path)
                    
            else:  # If it's a single video file path
                print(f"Processing video file: {input_path}")
                frame_paths = extract_video_frames_to_disk(
                    video_input=input_path, 
                    output_dir=temp_frames_dir, 
                    max_size=max_size,
                    use_all_frames=use_all_frames,
                    target_fps=target_fps
                )
        elif hasattr(input_path, "name"):  # File-like object (e.g., from Gradio upload)
            print(f"Processing uploaded video file: {input_path.name}")
            frame_paths = extract_video_frames_to_disk(
                video_input=input_path, 
                output_dir=temp_frames_dir, 
                max_size=max_size,
                use_all_frames=use_all_frames,
                target_fps=target_fps
            )
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        if not frame_paths:
            print("No frames extracted or read.")
            return []

        if use_all_frames:
            # For automatic mode, extract frames at consistent rate to ensure good overlap
            total_frames = len(frame_paths)
            
            # Try to get video duration to calculate fps-based sampling
            try:
                # Get video info using ffprobe
                if isinstance(input_path, (str, os.PathLike)) and not os.path.isdir(input_path):
                    video_path = str(input_path)
                elif hasattr(input_path, "name"):
                    video_path = input_path.name
                else:
                    video_path = None
                    
                if video_path and os.path.exists(video_path):
                    # Since ffmpeg already extracted frames at the target fps,
                    # we should use all extracted frames without further filtering
                    selected_frame_paths = frame_paths
                    print(f"Using all {len(selected_frame_paths)} frames extracted by ffmpeg at target fps {target_fps}")
                else:
                    # Fallback to using all frames if no video path
                    selected_frame_paths = frame_paths
                    print(f"Using all {len(selected_frame_paths)} frames")
            except Exception as e:
                print(f"Could not determine video duration ({e}), using frame-based sampling...")
                # Fallback: sample to approximate target_fps
                # Assume 30fps video if we don't know the actual fps
                assumed_fps = 30.0
                step = max(1, int(assumed_fps / target_fps))
                
                # For a 2-minute video at 3fps, we should get ~360 frames
                # Don't artificially limit the frames
                selected_frame_paths = frame_paths[::step]
                print(f"Sampled {len(selected_frame_paths)} frames (every {step} frames from {total_frames} total, targeting ~{target_fps} fps)")
        else:
            # Original behavior: Score and select optimal frames
            print(f"Scoring {len(frame_paths)} frames...")
            frames_scores = preprocess_frame_paths(frame_paths)
            
            # Select optimal frames
            selected_frames_indices = select_optimal_frames(
                scores=frames_scores, k=min(num_ref_views, len(frame_paths))
            )
            
            # Get paths to selected frames
            selected_frame_paths = [frame_paths[idx] for idx in selected_frames_indices]
            
            print(f"Selected {len(selected_frame_paths)} optimal frames out of {len(frame_paths)}")

        # Debug output
        print(f"DEBUG: use_all_frames={use_all_frames}, total extracted frames={len(frame_paths)}, selected frames={len(selected_frame_paths)}")
        
        # Copy selected frames to scene directory
        copy_selected_frames_to_scene_dir(selected_frame_paths, output_dir)
        
        # Return empty list since we're not loading frames into memory anymore
        # The actual frames are saved to disk in the scene directory
        return []
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
            print(f"Cleaned up temporary frame directory: {temp_frames_dir}")


def process_input_for_colmap_legacy(input_path, num_ref_views, output_dir, max_size=1024):
    """
    DEPRECATED: Original memory-intensive version.
    Helper function to read frames from video or image folder, select optimal ones,
    and save them to the output_dir/images.
    This is based on process_input from gradio_demo.py.
    Renamed to avoid potential confusion if 'process_input' is too generic.
    """
    frames_to_save_in_scene_dir = []
    if isinstance(input_path, (str, os.PathLike)):  # If input_path is a path string
        if os.path.isdir(input_path):  # If it's a directory of images
            print(f"Processing image directory: {input_path}")
            raw_frames = []
            image_files = sorted(
                [
                    f
                    for f in os.listdir(input_path)
                    if f.lower().endswith(("jpg", "jpeg", "png"))
                ]
            )
            for img_file in image_files:
                img = Image.open(os.path.join(input_path, img_file)).convert("RGB")
                # Resize if necessary, similar to video frames
                width, height = img.size
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                raw_frames.append(np.array(img))
        else:  # If it's a single video file path
            print(f"Processing video file: {input_path}")
            raw_frames = read_video_frames(video_input=input_path, max_size=max_size)
    elif hasattr(
        input_path, "name"
    ):  # If input_path is a file-like object (e.g., from Gradio upload)
        print(f"Processing uploaded video file: {input_path.name}")
        raw_frames = read_video_frames(video_input=input_path.name, max_size=max_size)
    else:
        raise ValueError(f"Unsupported input_path type: {type(input_path)}")

    if not raw_frames:
        print("No frames extracted or read.")
        return []

    frames_scores = preprocess_frames(
        raw_frames
    )  # Assuming preprocess_frames takes list of numpy arrays
    selected_frames_indices = select_optimal_frames(
        scores=frames_scores, k=min(num_ref_views, len(raw_frames))
    )
    frames_to_save_in_scene_dir = [
        raw_frames[frame_idx] for frame_idx in selected_frames_indices
    ]

    # The 'output_dir' here is the scene_dir where 'images' subfolder will be created
    save_frames_to_scene_dir(frames=frames_to_save_in_scene_dir, scene_dir=output_dir)
    return frames_to_save_in_scene_dir  # Returns the list of selected frame data (numpy arrays)


def orchestrate_video_to_colmap_scene(
    input_path,
    num_ref_views,
    max_size=1024,
    base_work_dir="../outputs/processed_scenes",
    use_automatic_mode=False,
    target_fps=3.0,
    colmap_config="low_memory",
):
    """
    Orchestrates the full video/image folder preprocessing pipeline:
    1. Creates a temporary scene directory.
    2. Reads frames, selects optimal ones, saves them.
    3. Runs COLMAP on the scene.
    Args:
        input_path (str or file-like): Path string, a Gradio file object, or a list (e.g., from gr.Examples).
        num_ref_views (int): Number of reference views to select.
        max_size (int): Maximum size for width or height after resizing.
        base_work_dir (str): Base directory for temporary scene directories.
        use_automatic_mode (bool): If True, use automatic_reconstructor-like settings.
    Returns:
        the list of selected frame image data and the path to the COLMAP processed scene directory.
        This is based on preprocess_input from gradio_demo.py.
    """
    actual_input_path_str = None
    input_name_part = "temp_scene"  # Default

    if hasattr(input_path, "name") and isinstance(
        input_path.name, str
    ):  # Gradio file object
        actual_input_path_str = input_path.name
        input_name_part = os.path.splitext(os.path.basename(input_path.name))[0]
    elif isinstance(input_path, (str, os.PathLike)):  # Direct path string
        actual_input_path_str = str(input_path)
        input_name_part = os.path.splitext(os.path.basename(input_path))[0]
    elif (
        isinstance(input_path, list) and input_path
    ):  # Handle list: take the first item.
        # gr.Examples often wraps the path in another list, e.g., [['path/to/example.mp4']]
        # So, we might need to unwrap it.
        first_item_candidate = input_path[0]
        if (
            isinstance(first_item_candidate, list) and first_item_candidate
        ):  # Check for nested list
            first_item = first_item_candidate[0]
        else:
            first_item = first_item_candidate

        if hasattr(first_item, "name") and isinstance(
            first_item.name, str
        ):  # Gradio file object in list
            actual_input_path_str = first_item.name
            input_name_part = os.path.splitext(os.path.basename(first_item.name))[0]
        elif isinstance(first_item, (str, os.PathLike)):  # Path string in list
            actual_input_path_str = str(first_item)
            input_name_part = os.path.splitext(os.path.basename(first_item))[0]
        else:
            print(f"Warning: Unsupported item type in input list: {type(first_item)}")
            return [], None
    else:
        print(f"Error: Unsupported input_path type: {type(input_path)}")
        return [], None

    if not actual_input_path_str:
        print("Error: Could not determine a valid input file path.")
        return [], None

    print(f"Orchestrating COLMAP scene from: {actual_input_path_str}")

    # Using a structured output directory instead of pure tempfile.mkdtemp for easier inspection
    # scene_dir_parent = tempfile.mkdtemp() # Original approach

    # Ensure base_work_dir exists
    os.makedirs(base_work_dir, exist_ok=True)
    
    # Use base_work_dir directly if it's been explicitly specified
    # This avoids adding video name or counter suffixes
    scene_dir = base_work_dir
    os.makedirs(scene_dir, exist_ok=True)
    print(f"Created scene directory for COLMAP: {scene_dir}")

    # Process video/images to extract and select optimal frames
    selected_frames_data = process_input_for_colmap(
        actual_input_path_str, num_ref_views, scene_dir, max_size, 
        use_all_frames=use_automatic_mode, target_fps=target_fps
    )
    
    # Check if images were saved to scene directory
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Frame processing failed for {input_path}. No images found in {images_dir}. Aborting COLMAP.")
        return [], None

    # Run COLMAP with PINHOLE camera model enforced
    run_colmap_on_scene(scene_dir, force_pinhole=True, colmap_config=colmap_config)

    print(f"COLMAP processing complete for {scene_dir}")
    return selected_frames_data, scene_dir


def get_reconstruction_size(recon_path):
    """Get the size (number of images) of a reconstruction."""
    try:
        reconstruction = pycolmap.Reconstruction(recon_path)
        return len(reconstruction.images)
    except:
        return 0


def merge_colmap_reconstructions(reconstructions_list, sparse_path):
    """
    Merge multiple COLMAP reconstructions into a single one.
    
    Args:
        reconstructions_list: List of (index, path) tuples
        sparse_path: Path to sparse directory
    
    Returns:
        Merged pycolmap.Reconstruction object
    """
    print(f"🔄 Merging {len(reconstructions_list)} reconstructions...")
    
    # Load all reconstructions
    reconstructions = []
    total_images = 0
    total_points = 0
    
    for idx, recon_path in reconstructions_list:
        try:
            recon = pycolmap.Reconstruction(recon_path)
            if len(recon.images) > 0:  # Only include non-empty reconstructions
                reconstructions.append((idx, recon))
                total_images += len(recon.images)
                total_points += len(recon.points3D)
                print(f"  Reconstruction {idx}: {len(recon.images)} images, {len(recon.points3D)} points")
        except Exception as e:
            print(f"  ⚠️  Could not load reconstruction {idx}: {e}")
    
    if not reconstructions:
        raise RuntimeError("No valid reconstructions to merge")
    
    if len(reconstructions) == 1:
        print(f"Only one valid reconstruction found, using it directly")
        return reconstructions[0][1]
    
    # Start with the largest reconstruction as base
    base_idx, base_recon = max(reconstructions, key=lambda x: len(x[1].images))
    print(f"📊 Using reconstruction {base_idx} as base ({len(base_recon.images)} images)")
    
    # Create a new merged reconstruction starting from the base
    merged = pycolmap.Reconstruction()
    
    # Copy cameras from base reconstruction
    for camera_id, camera in base_recon.cameras.items():
        merged.add_camera(camera)
    
    # Keep track of image names to avoid duplicates
    added_image_names = set()
    image_id_offset = 0
    point_id_offset = 0
    
    # Add all reconstructions
    for recon_idx, recon in reconstructions:
        print(f"  Adding reconstruction {recon_idx}...")
        
        # Add cameras (skip if already exists)
        for camera_id, camera in recon.cameras.items():
            if camera_id not in merged.cameras:
                merged.add_camera(camera)
        
        # Add images with ID offset to avoid conflicts
        for image_id, image in recon.images.items():
            if image.name not in added_image_names:
                new_image_id = image_id + image_id_offset
                # Create new image with offset ID
                new_image = pycolmap.Image(
                    id=new_image_id,
                    name=image.name,
                    camera_id=image.camera_id,
                    qvec=image.qvec,
                    tvec=image.tvec
                )
                merged.add_image(new_image)
                added_image_names.add(image.name)
            else:
                print(f"    Skipping duplicate image: {image.name}")
        
        # Add 3D points with ID offset
        for point_id, point in recon.points3D.items():
            new_point_id = point_id + point_id_offset
            merged.add_point3D(new_point_id, point.xyz, point.track, point.color)
        
        # Update offsets for next reconstruction
        if recon.images:
            image_id_offset += max(recon.images.keys()) + 1
        if recon.points3D:
            point_id_offset += max(recon.points3D.keys()) + 1
    
    print(f"✅ Merged reconstruction: {len(merged.images)} images, {len(merged.points3D)} points")
    
    return merged
