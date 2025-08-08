# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDGS (Eliminating Densification for Gaussian Splatting) is a 3D reconstruction system that improves 3D Gaussian Splatting by eliminating the need for densification. It uses dense initialization from triangulated 2D correspondences across training image pairs, achieving faster convergence and higher rendering quality.

## Development Environment

The project runs in a Docker container with CUDA support and uses conda for Python environment management.

### Starting the Environment

**From Host Machine:**
```bash
# Start the Docker container
docker compose up -d

# Execute commands inside the container
docker compose exec edgs-app bash
```

**From DevContainer (VS Code Dev Containers):**
```bash
# Use the helper script that automatically handles volume mounting
./script/docker-compose-devcontainer.sh up -d

# Execute commands inside the container
./script/docker-compose-devcontainer.sh exec edgs-app bash
```

Note: When running inside a devcontainer, the standard `docker compose` command will not properly mount the host directories due to path resolution issues. Always use the `docker-compose-devcontainer.sh` script which automatically detects the host paths and configures the volume mounts correctly.

### Python Environment
Once inside the container, the conda environment `edgs` is activated automatically. All Python commands should be run with this environment active.

## Common Development Commands

### Training Models

**Standard training from COLMAP scene:**
```bash
python script/train.py \
  train.gs_epochs=30000 \
  train.no_densify=True \
  gs.dataset.source_path=<scene_folder> \
  gs.dataset.model_path=<output_folder> \
  init_wC.matches_per_ref=20000 \
  init_wC.nns_per_ref=3 \
  init_wC.num_refs=180
```

**Process video to 3D model (full pipeline):**
```bash
# Standard processing
python script/fit_model_to_scene_full.py --video_path <video.mp4> --output_path <output_dir>

# With memory optimization (for low GPU memory)
python script/fit_model_to_scene_full.py --video_path <video.mp4> --config train_low_memory

# With specific COLMAP settings
python script/fit_model_to_scene_full.py --video_path <video.mp4> --colmap_config low_memory
```

### Running Interactive Tools

**Gradio web interface:**
```bash
python script/gradio_demo.py --port 7862
```

**JupyterLab:**
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=notebooks
```

### Memory-Optimized Training Scripts

For different GPU memory constraints:
```bash
# Highest quality (16GB+ GPU)
./script/run_docker_01_highest_quality.sh <colmap_path> <output_path>

# High quality (12GB+ GPU)  
./script/run_docker_02_high_quality.sh <colmap_path> <output_path>

# Optimal quality (10GB+ GPU)
./script/run_docker_03_optimal_quality.sh <colmap_path> <output_path>

# Medium quality (8GB+ GPU)
./script/run_docker_04_medium_quality.sh <colmap_path> <output_path>

# Low quality (6GB+ GPU)
./script/run_docker_05_low_quality.sh <colmap_path> <output_path>

# Lowest quality (4GB+ GPU)
./script/run_docker_06_lowest_quality.sh <colmap_path> <output_path>
```

### Evaluation
```bash
python script/full_eval.py -m360 <mipnerf360_folder> -tat <tanks_temples_folder> -db <deep_blending_folder>
```

## Architecture

### Core Components

**Training Pipeline (`source/`):**
- `trainer.py`: Main EDGSTrainer class orchestrating the training process
- `corr_init.py`: Dense initialization from 2D correspondences using RoMa matcher
- `utils_preprocess.py`: Video processing and COLMAP scene generation
- `networks.py`: Neural network components for exposure compensation
- `losses.py`: Custom loss functions for training
- `visualization.py`: Rendering and visualization utilities

**Configuration System (`configs/`):**
- Uses Hydra for configuration management
- Training configs: `train_*.yaml` files with different quality/memory profiles
- COLMAP configs: `colmap_*.yaml` for reconstruction settings
- Base GS config: `gs/base.yaml`

**Scripts (`script/`):**
- `train.py`: Main training entry point
- `fit_model_to_scene_full.py`: End-to-end pipeline from video/images to 3D model
- `run_docker_*.sh`: Memory-optimized training launchers
- `gradio_demo.py`: Interactive web interface

### Key Processing Flow

1. **Video/Image Input** → Frame extraction (ffmpeg/OpenCV)
2. **COLMAP Processing** → Camera poses and sparse point cloud
3. **Dense Initialization** → RoMa-based 2D correspondence matching
4. **Gaussian Splatting Training** → EDGS optimization without densification
5. **Model Output** → Saved checkpoints, rendered views, and metrics

### External Dependencies

- **Submodules:**
  - `submodules/gaussian-splatting/`: Base 3DGS implementation
  - `submodules/RoMa/`: Robust matcher for correspondence extraction

- **Key Python Packages:**
  - PyTorch with CUDA support
  - PyColmap for Structure-from-Motion
  - Hydra for configuration
  - Gradio for web interface
  - Wandb for experiment tracking (optional)

### Configuration Parameters

**Important training parameters:**
- `train.gs_epochs`: Number of training iterations
- `init_wC.matches_per_ref`: Correspondence points per reference view
- `init_wC.num_refs`: Number of reference views for initialization
- `train.no_densify`: Always True for EDGS (disables densification)

**Video processing parameters:**
- `--target_fps`: Frame extraction rate (default: 3.0)
- `--max_image_size`: Maximum image dimension for resizing (-1 for original)
- `--colmap_config`: COLMAP quality profile (high_accuracy/balanced/low_memory/very_low_memory)

## Data Paths

- Input data: `/EDGS/data/` (mapped from `./data/`)
- Output models: `/EDGS/outputs/` (mapped from `./outputs/`)
- Scripts: `/EDGS/script/` (mapped from `./script/`)
- Source code: `/EDGS/source/` (mapped from `./source/`)
- Notebooks: `/EDGS/notebooks/` (mapped from `./notebooks/`)

## GPU Memory Management

The system provides multiple configuration profiles for different GPU memory constraints:
- 16GB+: Use `train_01_highest_quality.yaml` or `train_02_high_quality.yaml`
- 8-12GB: Use `train_03_optimal_quality.yaml` or `train_04_medium_quality.yaml`
- 4-8GB: Use `train_05_low_quality.yaml` or `train_06_lowest_quality.yaml`

For COLMAP processing with limited memory, use:
- `--colmap_config very_low_memory` for 4-6GB systems
- `--colmap_config low_memory` for 6-8GB systems
- `--colmap_config balanced` for 8-12GB systems
- `--colmap_config high_accuracy` for 16GB+ systems