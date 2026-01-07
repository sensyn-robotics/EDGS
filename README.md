<h1 align="center">EDGS: Eliminating Densification for Efficient Convergence of 3DGS</h2>

<p align="center">
  <a href="https://www.linkedin.com/in/dmitry-kotovenko-dl/">Dmytro Kotovenko</a><sup>*</sup> ·
  <a href="https://www.linkedin.com/in/grebenkovao/">Olga Grebenkova</a><sup>*</sup> ·
  <a href="https://ommer-lab.com/people/ommer/">Björn Ommer</a>
</p>

<p align="center">CompVis @ LMU Munich · Munich Center for Machine Learning (MCML) </p>
<p align="center">* equal contribution </p>

<p align="center">
  <a href="https://compvis.github.io/EDGS/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2504.13204"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" alt="Paper"></a>
  <a href="https://colab.research.google.com/github/CompVis/EDGS/blob/main/notebooks/fit_model_to_scene_full.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://huggingface.co/spaces/CompVis/EDGS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face"></a>
  
</p>

<p align="center">
  <img src="./assets/Teaser2.png" width="99%">
</p>

<p>
<strong>3DGS</strong> initializes with a sparse set of Gaussians and progressively adds more in under-reconstructed regions. In contrast, <strong>EDGS</strong> starts with
a dense initialization from triangulated 2D correspondences across training image pairs, 
requiring only minimal refinement. This leads to <strong>faster convergence</strong> and <strong>higher rendering quality</strong>. Our method reaches the original 3DGS <strong>LPIPS score in just 25% of the training time</strong> and uses only <strong>60% of the splats</strong>. 
Renderings become <strong>nearly indistinguishable from ground truth after only 3,000 steps — without any densification</strong>.
</p>

<h3 align="center">3D scene reconstruction using our method in 11 seconds.</h3>
<p align="center">
  <img src="assets/video_fruits_our_optimization.gif" width="480" alt="3D Reconstruction Demo">
</p>



## 📚 Table of Contents
- [🚀 Quickstart](#sec-quickstart)
- [🛠️ Installation](#sec-install)
- [📦 Data](#sec-data)
- [🎬 Video Processing Improvements](#video-processing-improvements)
- [🏋️ Training](#sec-training)
- [🏗️ Reusing Our Model](#sec-reuse)
- [🛠️ Tools](#sec-tools)
- [📄 Citation](#sec-citation)

<a id="sec-quickstart"></a>
## 🚀 Quickstart
The fastest way to try our model is through the [Hugging Face demo](https://huggingface.co/spaces/magistrkoljan/EDGS), which lets you upload images or a video and interactively rotate the resulting 3D scene. For broad accessibility, we currently support only **forward-facing scenes**.
#### Steps:
1. Upload a list of photos or a single video.
2. Click **📸 Preprocess Input** to estimate 3D positions using COLMAP.
3. Click **🚀 Start Reconstruction** to run the model.

You can also **explore the reconstructed scene in 3D** directly in the browser.

> ⚡ Runtime: EDGS typically takes just **10–20 seconds**, plus **5–10 seconds** for COLMAP processing. Additional time may be needed to save outputs (model, video, 3D preview).

You can also run the same app locally on your machine with command: 
```CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --port 7862 --no_share```
Without `--no_share` flag you will get the adress for gradio app that you can share with the others allowing others to process their data on your server. 

Alternatively, check our [Colab notebook](https://colab.research.google.com/github/CompVis/EDGS/blob/main/notebooks/fit_model_to_scene_full.ipynb).




<a id="sec-install"></a>
## 🛠️ Installation

```bash
# Using Docker (recommended)
docker compose up -d

# Or manual installation
bash script/install.sh
```

<a id="sec-data"></a>
## 📦 Data

We evaluated on the following datasets:

- **MipNeRF360** — download [here](https://jonbarron.info/mipnerf360/). Unzip "Dataset Pt. 1" and "Dataset Pt. 2", then merge scenes.
- **Tanks & Temples + Deep Blending** — from the [original 3DGS repo](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

### Using Your Own Dataset

#### Option A
Use gradle demo.
After running `docker compose up -d`,
```
docker compose exec edgs-app bash
python script/gradio_demo.py --port 7862
```

#### Option B
From command line - Complete command with all possible arguments:
```bash
docker compose exec edgs-app python script/fit_model_to_scene_full.py \
    --input <input>                       # Path to input: COLMAP scene, image directory, video directory, or video file \
    --360                                 # Enable 360° equirectangular video processing (optional) \
    --colmap_config <colmap_preset>       # COLMAP preset (01=highest to 06=lowest quality) \
    --config <config_name>                # Training config (01=highest to 06=lowest quality) \
    --output_path <output_dir>            # Where to save EDGS model (optional)
```

**Available COLMAP configs (include video preprocessing settings):**
- `colmap_01_highest_quality`: target_fps=3.0, max_image_size=-1 (original resolution)
- `colmap_02_high_quality`: target_fps=3.0, max_image_size=2560
- `colmap_03_optimal_quality`: target_fps=3.0, max_image_size=1920 (default)
- `colmap_04_medium_quality`: target_fps=2.5, max_image_size=1600
- `colmap_05_low_quality`: target_fps=2.0, max_image_size=1024
- `colmap_06_lowest_quality`: target_fps=1.0, max_image_size=800

**Available training configs by GPU memory:**
- `train_01_highest_quality` (16GB+)
- `train_02_high_quality` (12GB+) 
- `train_03_optimal_quality` (10GB+)
- `train_04_medium_quality` (8GB+)
- `train_05_low_quality` (6GB+)
- `train_06_lowest_quality` (4GB+)

**Quick examples:**
```bash
# Simple usage
python script/fit_model_to_scene_full.py --input video.mp4

# 360° equirectangular video (converts to 5 cubemap faces: front, right, back, left, top)
python script/fit_model_to_scene_full.py --input video_360.mp4 --360

# Low memory system
python script/fit_model_to_scene_full.py --input video.mp4 \
    --config train_06_lowest_quality --colmap_config colmap_06_lowest_quality

# High quality
python script/fit_model_to_scene_full.py --input video.mp4 \
    --config train_02_high_quality --colmap_config colmap_02_high_quality

# 360° video with low memory settings
python script/fit_model_to_scene_full.py --input video_360.mp4 --360 \
    --config train_05_low_quality --colmap_config colmap_05_low_quality
```

**🔄 Resumable Pipeline:**

The `fit_model_to_scene_full.py` script now supports automatic checkpoint detection and resume functionality. Each stage (image extraction, COLMAP reconstruction, EDGS training) checks if it has already been completed before running:

- **Image Extraction**: If `images/` directory exists with frames, skips extraction
- **COLMAP Reconstruction**: If `sparse/` directory exists with valid scene data, skips COLMAP
- **EDGS Training**: If trained model exists (checkpoint or point cloud files), skips training

This means you can:
- ✅ Resume after interruptions without losing progress
- ✅ Re-run the same command safely - completed stages are skipped
- ✅ Save computation time by not redoing expensive operations

```bash
# First run - does everything
python script/fit_model_to_scene_full.py --input video.mp4 --output_path ./outputs/my_scene

# Second run - automatically skips completed stages
python script/fit_model_to_scene_full.py --input video.mp4 --output_path ./outputs/my_scene
```

To force retraining from scratch, delete the output directory first:
```bash
rm -rf ./outputs/my_scene
python script/fit_model_to_scene_full.py --input video.mp4 --output_path ./outputs/my_scene
```

#### Option C
Using Jupyter lab.
```
docker compose exec edgs-app bash
```
And in the terminal in the docker container,
```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=notebooks
```
After JupyterLab starts, it will print URLs to the terminal. Look for a URL containing a token, like:
    `http://127.0.0.1:8888/lab?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
Open `http://localhost:8888` (or `http://127.0.0.1:8888`) in your host browser.
When prompted for a "Password or token", paste the `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` part from the URL in step 4 into the field and log in. Alternatively, you can paste the full URL from step 4 directly into your browser.

#### Option D
You can use the same data format as the [3DGS project](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes). Please follow their guide to prepare your scene.

Expected folder structure:
```
scene_folder
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

```
docker compose exec edgs-app bash
```
Then run training command as described below section.

Nerf synthetic format is also acceptable. 

You can also use functions provided in our code to convert a collection of images or a sinlge video into a desired format. However, this may requre tweaking and processing time can be large for large collection of images with little overlap.


<a id="sec-training"></a>
## 🏋️ Training


To optimize on a single scene in COLMAP format use this code.  
```bash
python script/train.py \
  train.gs_epochs=30000 \
  train.no_densify=True \
  gs.dataset.source_path=<scene folder> \
  gs.dataset.model_path=<output folder> \
  init_wC.matches_per_ref=20000 \
  init_wC.nns_per_ref=3 \
  init_wC.num_refs=180
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>
  
  * `train.gs_epochs`
  Number of training iterations (steps) for Gaussian Splatting.
  * `train.no_densify`
  Disables densification. True by default.
  * `gs.dataset.source_path`
  Path to your input dataset directory. This should follow the same format as the original 3DGS dataset structure.
  * `gs.dataset.model_path`
  Output directory where the trained model, logs, and renderings will be saved.
  * `init_wC.matches_per_ref`
  Number of 2D feature correspondences to extract per reference view for initialization. More matches leads to more gaussians.
  * `init_wC.nns_per_ref`
  Number of nearest neighbor images used per reference during matching.
  * `init_wC.num_refs`
  Total number of reference views sampled. 
  * `wandb.mode`
    Specifies how Weights & Biases (W&B) logging is handled.

    - Default: `"disabled"`
    - Options:
      - `"online"` — log to the W&B server in real-time
      - `"offline"` — save logs locally to sync later
      - `"disabled"` — turn off W&B logging entirely

    If you want to enable W&B logging, make sure to also configure:

    - `wandb.project` — the name of your W&B project
    - `wandb.entity` — your W&B username or team name

Example override:
```bash
wandb.mode=online wandb.project=EDGS wandb.entity=your_username train.gs_epochs=15_000 init_wC.matches_per_ref=15_000
```
</details>
<br>

To run full evaluation on all datasets:

```bash
python script/full_eval.py -m360 <mipnerf360_folder> -tat <tanks_temples_folder> -db <deep_blending_folder>
```
<a id="sec-reuse"></a>
## 🏗️ Reusing Our Model

Our model is essentially a better **initialization module** for Gaussian Splatting. You can integrate it into your pipeline by calling:

```python
source.corr_init.init_gaussians_with_corr(...)
```
### Input arguments:
- A GaussianModel and Scene instance
- A configuration namespace `cfg.init_wC` to specify parameters like the number of matches, neighbors, and reference views
- A RoMA model (automatically instantiated if not provided)



<a id="sec-tools"></a>
## 🛠️ Tools

### Tree Diameter Estimator

Estimates tree trunk diameters using depth images from Gaussian Splatting rendering, tree segmentation results (COCO format), and camera focal length.

```bash
python script/tree_diameter_estimator.py <scene_path> [-o output.json] [-v] [--visualize DIR]
```

**Arguments:**
- `scene_path`: Path to the scene directory containing `cameras.json`, depth maps, and segmentation data
- `-o, --output`: Output JSON file for results (optional)
- `-v, --verbose`: Print detailed results
- `--visualize DIR`: Output directory for visualization images with segmentation overlays and diameter annotations

**Example:**
```bash
# Basic usage with verbose output
python script/tree_diameter_estimator.py ./outputs/forest_scene -o measurements.json -v

# Generate visualization images
python script/tree_diameter_estimator.py ./outputs/forest_scene --visualize ./vis_output
```

<a id="sec-citation"></a>
## 📄 Citation
```bibtex
@misc{kotovenko2025edgseliminatingdensificationefficient,
      title={EDGS: Eliminating Densification for Efficient Convergence of 3DGS}, 
      author={Dmytro Kotovenko and Olga Grebenkova and Björn Ommer},
      year={2025},
      eprint={2504.13204},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2504.13204}, 
}
```



