#!/usr/bin/env python
"""Quick PSNR evaluation for existing Gaussian Splatting models.
Loads the model from checkpoint and evaluates on test set without training.

Usage:
    python script/quick_eval.py --model_path outputs/scene_name
"""

import sys
sys.path.append('submodules/gaussian-splatting')

import os
import torch
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def evaluate_model(model_path: str, iteration: int = 30000, sh_degree: int = None):
    """Evaluate PSNR/SSIM/LPIPS for a trained model."""

    print(f"\n=== Evaluating: {model_path} ===")

    # Check if checkpoint exists
    ckpt_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")

    # Try to read sh_degree from train_config.yaml
    if sh_degree is None:
        config_path = os.path.join(model_path, "train_config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
            sh_degree = config.get('gs', {}).get('sh_degree', 3)
            print(f"  Using sh_degree={sh_degree} from config")
        else:
            sh_degree = 3
            print(f"  Using default sh_degree={sh_degree}")

    if not os.path.exists(ply_path):
        print(f"ERROR: PLY file not found: {ply_path}")
        return None

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Create a minimal config for loading
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

    # Check if source path exists
    if not os.path.exists(args.source_path):
        print(f"ERROR: Source path not found: {args.source_path}")
        return None

    # Load scene and gaussians
    print("Loading scene...")
    try:
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        import traceback
        print(f"ERROR loading scene: {e}")
        traceback.print_exc()
        return None

    # Get test cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("WARNING: No test cameras found, using train cameras")
        test_cameras = scene.getTrainCameras()

    print(f"Evaluating on {len(test_cameras)} test views...")

    # Setup rendering
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Pipeline config
    class PipelineConfig:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
        antialiasing = False

    pipe = PipelineConfig()

    # Compute metrics
    psnrs = []
    ssims = []
    lpipss = []

    with torch.no_grad():
        for viewpoint in tqdm(test_cameras, desc="Computing metrics"):
            # Render
            rendering = render(viewpoint, gaussians, pipe, background)["render"]
            gt = viewpoint.original_image[:3, :, :].cuda()

            # Add batch dimension for psnr/ssim computation
            rendering_batch = rendering.unsqueeze(0)
            gt_batch = gt.unsqueeze(0)

            # Compute metrics
            psnrs.append(psnr(rendering_batch, gt_batch).mean().item())
            ssims.append(ssim(rendering_batch, gt_batch).item())
            lpipss.append(lpips(rendering_batch, gt_batch, net_type='vgg').item())

    # Average metrics
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    avg_lpips = sum(lpipss) / len(lpipss)

    print(f"\nResults:")
    print(f"  PSNR:  {avg_psnr:.4f}")
    print(f"  SSIM:  {avg_ssim:.4f}")
    print(f"  LPIPS: {avg_lpips:.4f}")

    # Determine pass/fail
    status = "PASS" if avg_psnr >= 20.0 else "FAIL"
    print(f"  Status: {status} (target PSNR >= 20)")

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "status": status
    }


def main():
    parser = ArgumentParser(description="Quick PSNR evaluation")
    parser.add_argument("--model_path", "-m", required=True, help="Path to model directory")
    parser.add_argument("--iteration", "-i", type=int, default=30000, help="Iteration to evaluate")
    args = parser.parse_args()

    result = evaluate_model(args.model_path, args.iteration)

    if result:
        print(f"\n=== Final: PSNR={result['psnr']:.4f} ({result['status']}) ===")
        return 0 if result['status'] == 'PASS' else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
