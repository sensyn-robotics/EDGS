#!/usr/bin/env python
"""Visual progress checker for EDGS training pipeline.

Generates visual reports for:
  1. COLMAP reconstruction: camera positions + sparse 3D point cloud
  2. 3DGS intermediate state: rendered views vs ground truth

Usage (inside Docker container):
    # Check COLMAP reconstruction
    python script/check_progress.py --scene_path outputs/scene/video_scene

    # Check 3DGS training (auto-detects latest checkpoint)
    python script/check_progress.py --model_path outputs/scene

    # Check both
    python script/check_progress.py --model_path outputs/scene --scene_path outputs/scene/video_scene

    # Specify iteration
    python script/check_progress.py --model_path outputs/scene --iteration 7000
"""

import argparse
import glob
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "submodules", "gaussian-splatting"))


def find_colmap_scene(model_path):
    """Try to find the COLMAP scene directory from a model path."""
    candidates = [
        os.path.join(model_path, "video_scene"),
        os.path.join(model_path, "image_scene"),
        model_path,
    ]
    for c in candidates:
        sparse = os.path.join(c, "sparse", "0")
        if os.path.isdir(sparse):
            return c
    return None


def find_latest_checkpoint(model_path):
    """Find the latest saved iteration in a model directory."""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        return None
    iterations = []
    for d in os.listdir(pc_dir):
        if d.startswith("iteration_"):
            try:
                iterations.append(int(d.replace("iteration_", "")))
            except ValueError:
                pass
    return max(iterations) if iterations else None


def visualize_colmap(scene_path, output_dir):
    """Visualize COLMAP reconstruction: cameras + sparse points."""
    import pycolmap
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    sparse_path = os.path.join(scene_path, "sparse", "0")
    if not os.path.isdir(sparse_path):
        print(f"No sparse reconstruction at {sparse_path}")
        return

    print(f"Loading COLMAP reconstruction from {sparse_path}...")
    reconstruction = pycolmap.Reconstruction(sparse_path)

    num_cameras = len(reconstruction.cameras)
    num_images = len(reconstruction.images)
    num_points = len(reconstruction.points3D)
    print(f"  Cameras: {num_cameras}, Images: {num_images}, 3D Points: {num_points}")

    # Extract camera positions (world coords)
    cam_positions = []
    cam_forwards = []
    for image_id, image in reconstruction.images.items():
        # Camera center in world coordinates
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        center = -R.T @ t
        cam_positions.append(center)
        # Forward direction (z-axis of camera in world)
        forward = R.T @ np.array([0, 0, 1])
        cam_forwards.append(forward)

    cam_positions = np.array(cam_positions)
    cam_forwards = np.array(cam_forwards)

    # Extract 3D points (subsample if too many)
    points = []
    colors = []
    for pid, p3d in reconstruction.points3D.items():
        points.append(p3d.xyz)
        colors.append(p3d.color / 255.0)

    points = np.array(points) if points else np.zeros((0, 3))
    colors = np.array(colors) if colors else np.zeros((0, 3))

    # Subsample points for plotting
    max_plot_points = 50000
    if len(points) > max_plot_points:
        idx = np.random.choice(len(points), max_plot_points, replace=False)
        points_plot = points[idx]
        colors_plot = colors[idx]
    else:
        points_plot = points
        colors_plot = colors

    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Top-down view (XZ plane) ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (dim1, dim2, title) in zip(axes, [
        (0, 2, "Top-down (XZ)"),
        (0, 1, "Front (XY)"),
        (2, 1, "Side (ZY)"),
    ]):
        if len(points_plot) > 0:
            ax.scatter(points_plot[:, dim1], points_plot[:, dim2],
                       c=colors_plot, s=0.3, alpha=0.5)
        ax.scatter(cam_positions[:, dim1], cam_positions[:, dim2],
                   c='red', s=20, marker='^', zorder=5, label='cameras')
        # Draw forward direction
        scale = np.percentile(np.abs(cam_positions), 90) * 0.05 if len(cam_positions) > 0 else 0.1
        for pos, fwd in zip(cam_positions, cam_forwards):
            ax.arrow(pos[dim1], pos[dim2],
                     fwd[dim1] * scale, fwd[dim2] * scale,
                     head_width=scale * 0.3, head_length=scale * 0.2,
                     fc='red', ec='red', alpha=0.6)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

    fig.suptitle(f"COLMAP: {num_images} images, {num_points} points, {num_cameras} camera(s)",
                 fontsize=14)
    plt.tight_layout()
    path_2d = os.path.join(output_dir, "colmap_views.png")
    plt.savefig(path_2d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_2d}")

    # --- Plot 2: 3D scatter with plotly (interactive HTML) ---
    try:
        import plotly.graph_objects as go

        traces = []
        if len(points_plot) > 0:
            rgb_str = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_plot]
            traces.append(go.Scatter3d(
                x=points_plot[:, 0], y=points_plot[:, 1], z=points_plot[:, 2],
                mode='markers',
                marker=dict(size=1, color=rgb_str, opacity=0.6),
                name=f'Points ({num_points})'
            ))

        traces.append(go.Scatter3d(
            x=cam_positions[:, 0], y=cam_positions[:, 1], z=cam_positions[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='diamond'),
            name=f'Cameras ({num_images})'
        ))

        # Camera forward directions as lines
        lines_x, lines_y, lines_z = [], [], []
        for pos, fwd in zip(cam_positions, cam_forwards):
            end = pos + fwd * scale * 3
            lines_x.extend([pos[0], end[0], None])
            lines_y.extend([pos[1], end[1], None])
            lines_z.extend([pos[2], end[2], None])

        traces.append(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='red', width=2),
            name='Camera dirs'
        ))

        fig3d = go.Figure(data=traces)
        fig3d.update_layout(
            title=f"COLMAP: {num_images} imgs, {num_points} pts",
            scene=dict(aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        path_html = os.path.join(output_dir, "colmap_3d.html")
        fig3d.write_html(path_html)
        print(f"  Saved: {path_html} (open in browser for interactive 3D)")
    except ImportError:
        print("  (plotly not available, skipping interactive 3D view)")

    # --- Plot 3: Sample input images ---
    images_dir = os.path.join(scene_path, "images")
    if os.path.isdir(images_dir):
        from PIL import Image
        all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                          glob.glob(os.path.join(images_dir, "*.jpg")))
        if all_imgs:
            n_samples = min(8, len(all_imgs))
            indices = np.linspace(0, len(all_imgs) - 1, n_samples, dtype=int)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for ax, idx in zip(axes.flat, indices):
                img = Image.open(all_imgs[idx])
                ax.imshow(img)
                ax.set_title(os.path.basename(all_imgs[idx]), fontsize=8)
                ax.axis('off')
            for ax in axes.flat[n_samples:]:
                ax.axis('off')
            plt.suptitle(f"Sample images ({len(all_imgs)} total)", fontsize=14)
            plt.tight_layout()
            path_samples = os.path.join(output_dir, "colmap_sample_images.png")
            plt.savefig(path_samples, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path_samples}")


def visualize_3dgs(model_path, iteration, output_dir):
    """Render 3DGS views at a given checkpoint and compare with ground truth."""
    import torch
    from matplotlib import pyplot as plt

    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"No checkpoint at iteration {iteration} ({ply_path})")
        return

    print(f"Loading 3DGS model at iteration {iteration}...")

    from scene import Scene, GaussianModel
    from gaussian_renderer import render

    # Minimal args for scene loading
    class Args:
        def __init__(self):
            self.source_path = find_colmap_scene(model_path) or model_path
            self.model_path = model_path
            self.images = "images"
            self.depths = ""
            self.resolution = 1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.sh_degree = 3
            self.train_test_exp = False

    args = Args()

    if not os.path.exists(args.source_path):
        print(f"  Source path not found: {args.source_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    except Exception as e:
        print(f"  Error loading scene: {e}")
        return

    num_gaussians = len(gaussians._xyz)
    print(f"  Loaded {num_gaussians} gaussians")

    # Pipeline config
    class PipelineConfig:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
        antialiasing = False

    pipe = PipelineConfig()
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Get cameras
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = test_cameras if len(test_cameras) > 0 else train_cameras

    n_views = min(8, len(cameras))
    indices = np.linspace(0, len(cameras) - 1, n_views, dtype=int)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(n_views, 2, figsize=(16, 4 * n_views))
    if n_views == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            viewpoint = cameras[idx]
            rendering = render(viewpoint, gaussians, pipe, bg)["render"]
            rendered_np = rendering.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            gt_np = viewpoint.original_image[:3].cpu().numpy().transpose(1, 2, 0)
            gt_np = np.clip(gt_np, 0, 1)

            axes[row, 0].imshow(gt_np)
            axes[row, 0].set_title(f"GT: {viewpoint.image_name}", fontsize=9)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(rendered_np)
            axes[row, 1].set_title(f"Rendered (iter {iteration})", fontsize=9)
            axes[row, 1].axis('off')

    plt.suptitle(f"3DGS @ iter {iteration} | {num_gaussians} gaussians | {len(cameras)} views",
                 fontsize=14)
    plt.tight_layout()
    path_render = os.path.join(output_dir, f"3dgs_iter_{iteration}.png")
    plt.savefig(path_render, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_render}")

    # Also save individual high-res comparisons for the first 4 views
    for i, idx in enumerate(indices[:4]):
        viewpoint = cameras[idx]
        with torch.no_grad():
            rendering = render(viewpoint, gaussians, pipe, bg)["render"]
        rendered_np = (rendering.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gt_np = (np.clip(viewpoint.original_image[:3].cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

        # Side by side
        combined = np.concatenate([gt_np, rendered_np], axis=1)
        from PIL import Image
        Image.fromarray(combined).save(os.path.join(output_dir, f"3dgs_compare_{i}_iter{iteration}.png"))

    print(f"  Saved {min(4, n_views)} high-res comparison images")

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Visual progress checker for EDGS pipeline")
    parser.add_argument("--model_path", "-m", type=str, default=None,
                        help="Path to EDGS model output directory")
    parser.add_argument("--scene_path", "-s", type=str, default=None,
                        help="Path to COLMAP scene directory (auto-detected from model_path if not given)")
    parser.add_argument("--iteration", "-i", type=int, default=None,
                        help="3DGS iteration to visualize (auto-detects latest)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Where to save visualizations (default: <model_path>/progress/)")
    args = parser.parse_args()

    if args.model_path is None and args.scene_path is None:
        parser.error("Provide at least --model_path or --scene_path")

    output_dir = args.output_dir
    if output_dir is None:
        base = args.model_path or args.scene_path
        output_dir = os.path.join(base, "progress")

    # --- COLMAP visualization ---
    scene_path = args.scene_path
    if scene_path is None and args.model_path:
        scene_path = find_colmap_scene(args.model_path)

    if scene_path:
        print("\n=== COLMAP Reconstruction ===")
        visualize_colmap(scene_path, output_dir)
    else:
        print("(No COLMAP scene found, skipping)")

    # --- 3DGS visualization ---
    if args.model_path:
        iteration = args.iteration or find_latest_checkpoint(args.model_path)
        if iteration:
            print(f"\n=== 3DGS Training (iteration {iteration}) ===")
            visualize_3dgs(args.model_path, iteration, output_dir)
        else:
            print("\n(No 3DGS checkpoints found yet, training may still be in COLMAP/init phase)")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
