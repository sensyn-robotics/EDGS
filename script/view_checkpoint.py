#!/usr/bin/env python
"""View 3DGS checkpoint during training - renders sample views to check quality.

Usage:
    python script/view_checkpoint.py --model_path outputs/scene_name [--iteration 7000]
"""

import sys
sys.path.append('submodules/gaussian-splatting')

import os
import torch
import yaml
from argparse import ArgumentParser
from PIL import Image
import numpy as np

from scene import Scene, GaussianModel
from gaussian_renderer import render


def find_latest_checkpoint(model_path):
    """Find the latest available checkpoint iteration."""
    ply_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(ply_dir):
        return None

    iterations = []
    for d in os.listdir(ply_dir):
        if d.startswith("iteration_"):
            try:
                iterations.append(int(d.split("_")[1]))
            except:
                pass

    return max(iterations) if iterations else None


def render_views(model_path, iteration=None, num_views=4, output_dir=None):
    """Render sample views from the model."""

    if iteration is None:
        iteration = find_latest_checkpoint(model_path)
        if iteration is None:
            print("No checkpoint found yet. Training may still be in COLMAP phase.")
            return None

    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Checkpoint not found: {ply_path}")
        return None

    print(f"Loading checkpoint: iteration {iteration}")

    # Get sh_degree from config
    config_path = os.path.join(model_path, "train_config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        sh_degree = config.get('gs', {}).get('sh_degree', 3)
    else:
        sh_degree = 3

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    class Args:
        def __init__(self):
            self.source_path = os.path.join(model_path, "video_scene")
            self.model_path = model_path
            self.images = "images"
            self.depths = ""
            self.resolution = 1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
            self.sh_degree = sh_degree
            self.train_test_exp = False

    args = Args()

    # Load model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Get cameras
    cameras = scene.getTrainCameras()
    if len(cameras) == 0:
        print("No cameras found")
        return None

    # Select evenly spaced views
    step = max(1, len(cameras) // num_views)
    selected_cameras = [cameras[i * step] for i in range(min(num_views, len(cameras)))]

    # Render
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    class PipelineConfig:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
        antialiasing = False

    pipe = PipelineConfig()

    rendered_images = []
    with torch.no_grad():
        for i, cam in enumerate(selected_cameras):
            rendering = render(cam, gaussians, pipe, bg_color)["render"]
            img = rendering.cpu().numpy().transpose(1, 2, 0)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            rendered_images.append(img)
            print(f"  Rendered view {i+1}/{len(selected_cameras)}")

    # Create grid image
    if len(rendered_images) > 0:
        h, w = rendered_images[0].shape[:2]
        cols = 2
        rows = (len(rendered_images) + cols - 1) // cols
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for i, img in enumerate(rendered_images):
            r, c = i // cols, i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = img

        # Save
        if output_dir is None:
            output_dir = model_path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"preview_iter{iteration}.png")
        Image.fromarray(grid).save(output_path)
        print(f"\nSaved preview: {output_path}")
        return output_path

    return None


def main():
    parser = ArgumentParser(description="View 3DGS checkpoint during training")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model directory")
    parser.add_argument("--iteration", "-i", type=int, default=None, help="Iteration to render (default: latest)")
    parser.add_argument("--num_views", "-n", type=int, default=4, help="Number of views to render")
    parser.add_argument("--output_dir", "-o", default=None, help="Output directory for preview")
    args = parser.parse_args()

    result = render_views(args.model_path, args.iteration, args.num_views, args.output_dir)
    if result:
        print(f"\nOpen the preview image to check training progress:")
        print(f"  xdg-open {result}")


if __name__ == "__main__":
    main()
