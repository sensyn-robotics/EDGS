#!/usr/bin/env python
#
# Render depth maps from a trained Gaussian Splatting model.
# Adapted for EDGS repository structure.
#

import argparse
import os
import sys

# Add paths for EDGS structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
gs_path = os.path.join(project_root, "submodules", "gaussian-splatting")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if gs_path not in sys.path:
    sys.path.insert(0, gs_path)

import math
import numpy as np
import torch
import torchvision
from plyfile import PlyData
from tqdm import tqdm

from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state


def detect_sh_degree_from_ply(ply_path: str) -> int:
    """Detect the spherical harmonics degree from a PLY file."""
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    # Number of extra features = 3 * (sh_degree + 1)^2 - 3
    # So: (num_extra + 3) / 3 = (sh_degree + 1)^2
    # sh_degree = sqrt((num_extra + 3) / 3) - 1
    num_extra = len(extra_f_names)
    sh_degree = int(math.sqrt((num_extra + 3) / 3)) - 1
    return max(0, sh_degree)

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, output_path=None):
    if output_path:
        # If explicit output path is given, everything goes there under train/test
        base_path = os.path.join(output_path, name, "ours_{}".format(iteration))
    else:
        # Legacy behavior
        base_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    depth_path = os.path.join(base_path, "depth")
    depth_vis_path = os.path.join(base_path, "depth_vis")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(depth_vis_path, exist_ok=True)

    # Sort views by image_name to ensure consistent ordering
    sorted_views = sorted(views, key=lambda v: v.image_name)

    for idx, view in enumerate(tqdm(sorted_views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = result["render"]
        depth = result["depth"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            depth = depth[..., depth.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # Use original image name (without extension) for consistent correspondence
        image_name = os.path.splitext(view.image_name)[0]

        torchvision.utils.save_image(rendering, os.path.join(render_path, image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, image_name + ".png"))

        # Save Raw Depth as .npy
        np.save(os.path.join(depth_path, image_name + ".npy"), depth.cpu().numpy())

        # Save Visualization (Normalize for visibility, simple gray scale)
        depth_vis = depth.clone().detach()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        torchvision.utils.save_image(depth_vis, os.path.join(depth_vis_path, image_name + ".png"))


def render_sets(dataset, iteration: int, pipeline, skip_train: bool, skip_test: bool, separate_sh: bool, ply_path: str = None, output_path: str = None):
    with torch.no_grad():
        # If custom ply_path is provided, detect SH degree from the file
        if ply_path:
            detected_sh = detect_sh_degree_from_ply(ply_path)
            print(f"Detected SH degree from PLY: {detected_sh}")
            gaussians = GaussianModel(detected_sh)
            # Load scene without loading the model (iteration=None)
            scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
            # Load the custom PLY file
            print(f"Loading custom PLY: {ply_path}")
            gaussians.load_ply(ply_path, dataset.train_test_exp)
            # Use a reasonable iteration number for output path
            loaded_iter = iteration if iteration > 0 else 30000
        else:
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            loaded_iter = scene.loaded_iter if hasattr(scene, 'loaded_iter') else iteration

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, output_path=output_path)

        if not skip_test:
            render_set(dataset.model_path, "test", loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Render depth maps from a trained Gaussian Splatting model.

Required parameters:
  --source_path, -s    Path to the dataset (COLMAP scene with cameras.bin/txt)
  --model_path, -m     Path to the trained model directory (contains cfg_args and point_cloud/)

Usage modes:
  1. Standard mode (from trained model):
     python script/render_depth.py -m <model_path> -s <source_path>

  2. Explicit PLY mode (custom ply file):
     python script/render_depth.py -s <source_path> --ply_file <path/to/point_cloud.ply> --output_path <output_dir>

Examples:
  python script/render_depth.py -m outputs/my_scene -s data/my_scene
  python script/render_depth.py -s data/my_scene --ply_file custom.ply --output_path renders/
""", formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model parameters
    parser.add_argument("--source_path", "-s", type=str, required=True, help="Path to the dataset (COLMAP scene)")
    parser.add_argument("--model_path", "-m", type=str, default=None, help="Path to the trained model directory")
    parser.add_argument("--images", type=str, default="images", help="Images folder name")
    parser.add_argument("--resolution", type=int, default=-1, help="Resolution to load images at")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--data_device", type=str, default="cuda", help="Device for data loading")
    parser.add_argument("--eval", action="store_true", help="Use eval mode (split train/test)")

    # Pipeline parameters
    parser.add_argument("--convert_SHs_python", action="store_true", help="Convert SHs in Python")
    parser.add_argument("--compute_cov3D_python", action="store_true", help="Compute 3D covariance in Python")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--antialiasing", action="store_true", help="Enable antialiasing")

    # Rendering options
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load (default: -1 for latest)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering train set")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test set")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--train_test_exp", action="store_true", help="Train/test exposure mode")

    # Explicit path arguments
    parser.add_argument("--ply_file", type=str, default=None, help="Explicit path to point_cloud.ply")
    parser.add_argument("--output_path", type=str, default=None, help="Explicit output directory")

    args = parser.parse_args()

    # Handle explicit PLY mode - set model_path from output_path if not provided
    if args.ply_file and args.output_path and not args.model_path:
        args.model_path = args.output_path
        os.makedirs(args.model_path, exist_ok=True)

    # Try to load cfg_args if available
    if args.model_path:
        cfgfilepath = os.path.join(args.model_path, "cfg_args")
        if os.path.exists(cfgfilepath):
            print("Loading config from", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                cfgfile_string = cfg_file.read()
                args_cfgfile = eval(cfgfile_string)

                # Merge: command line args override config file
                for k, v in vars(args_cfgfile).items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
        else:
            print(f"Config file not found at {cfgfilepath}, using command line arguments.")

    # Ensure source_path is absolute
    args.source_path = os.path.abspath(args.source_path)

    print(f"Rendering depth maps")
    print(f"  Source path: {args.source_path}")
    print(f"  Model path: {args.model_path}")
    if args.ply_file:
        print(f"  PLY file: {args.ply_file}")
    if args.output_path:
        print(f"  Output path: {args.output_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Create ModelParams-like namespace
    class ModelParamsSimple:
        def __init__(self, args):
            self.sh_degree = args.sh_degree
            self.source_path = args.source_path
            self.model_path = args.model_path
            self.images = args.images
            self.depths = ""  # Required by Scene but not used for rendering
            self.resolution = args.resolution
            self.white_background = args.white_background
            self.data_device = args.data_device
            self.eval = args.eval
            self.train_test_exp = getattr(args, 'train_test_exp', False)

    # Create PipelineParams-like namespace
    class PipelineParamsSimple:
        def __init__(self, args):
            self.convert_SHs_python = args.convert_SHs_python
            self.compute_cov3D_python = args.compute_cov3D_python
            self.debug = args.debug
            self.antialiasing = args.antialiasing

    model_params = ModelParamsSimple(args)
    pipeline_params = PipelineParamsSimple(args)

    render_sets(
        model_params,
        args.iteration,
        pipeline_params,
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        ply_path=args.ply_file,
        output_path=args.output_path
    )

    print("Done!")
