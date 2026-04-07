# EDGS Configuration Guide

This directory contains training and COLMAP configurations for EDGS (Eliminating Densification for Gaussian Splatting).

## Config Structure

```
configs/
├── gs/base.yaml              # Base Gaussian Splatting defaults (inherited by all train configs)
├── train.yaml                # Base EDGS training defaults (inherited by train_01-06)
├── train_01_highest_quality.yaml  # ⭐⭐⭐⭐⭐ A100 — max quality
├── train_02_high_quality.yaml     # ⭐⭐⭐⭐   12-16GB — progressive densification
├── train_03_optimal_quality.yaml  # ⭐⭐⭐     12-16GB — balanced
├── train_04_medium_quality.yaml   # ⭐⭐⭐     8-12GB — memory-conscious
├── train_05_low_quality.yaml      # ⭐⭐       8-12GB — SfM-only init
├── train_06_lowest_quality.yaml   # ⭐         4-8GB — minimal
├── colmap_01_highest_quality.yaml # Original resolution, strictest matching
├── colmap_02_high_quality.yaml    # 2.5K, high feature count
├── colmap_03_optimal_quality.yaml # 1080p, COLMAP defaults (default)
├── colmap_04_medium_quality.yaml  # 1600px, moderate features
├── colmap_05_low_quality.yaml     # 1024px, reduced features
├── colmap_06_lowest_quality.yaml  # 1024px, minimal features
├── colmap_07_360_optimized.yaml   # 360 video, 3fps, 5 perspective views/frame
└── colmap_08_360_2fps.yaml        # 360 video, 2fps variant
```

### Inheritance

```
gs/base.yaml  ←  train.yaml  ←  train_01..06.yaml
```

All numbered train configs inherit from `train.yaml` via Hydra `defaults: [train]`, and `train.yaml` inherits from `gs/base.yaml` via `defaults: [gs: base]`. COLMAP configs are standalone (no inheritance).

## Training Configs

### Key parameters that differ across tiers

| Config | GPU | Epochs | Densify | EDGS Init | matches/ref | num_refs | nns | SH | Batch | SfM Init |
|--------|-----|--------|---------|-----------|-------------|----------|-----|----|-------|----------|
| 01 highest | A100 | 150k | Yes | Yes | 20,000 | 240 | 5 | 3 | 8 | Yes |
| 02 high | 12-16GB | 100k | Yes | Yes | 12,000 | 150 | 3 | 1 | 1 | Yes |
| 03 optimal | 12-16GB | 90k | Yes | Yes | 15,000 | 180 | 3 | 3 | 12 | No |
| 04 medium | 8-12GB | 60k | Yes | Yes | 6,000 | 120 | 2 | 1 | 1 | Yes |
| 05 low | 8-12GB | 45k | No | **No** | — | — | — | 1 | 1 | Yes |
| 06 lowest | 4-8GB | 30k | No | **No** | — | — | — | 0 | 1 | Yes |

**Notes:**
- **EDGS Init** (`init_wC.use`): The core EDGS feature — dense initialization from RoMa correspondences. Disabled in 05/06 to avoid OOM on smaller GPUs.
- **Densify** (`train.no_densify`): Standard 3DGS densification. Disabled in 05/06 for consistent memory usage.
- **SfM Init** (`init_wC.add_SfM_init`): Include COLMAP sparse points alongside EDGS correlation points.
- **SH degree**: 0 = DC only (flat color), 1 = basic view-dependence, 3 = full view-dependent effects.

### train.yaml (base config)

The parent config for all numbered tiers. Key defaults:
- `gs_epochs: 30000`, `no_densify: False`, `matches_per_ref: 20000`, `num_refs: 360`
- Represents the original EDGS paper configuration
- Can be used directly via `--config train` but the numbered tiers are preferred

## COLMAP Configs

Control video preprocessing (frame extraction) and COLMAP reconstruction quality.

| Config | Target GPU | FPS | Image Size | SIFT Features | Matching Strictness |
|--------|-----------|-----|------------|---------------|---------------------|
| 01 highest | 16GB+ | 3.0 | original (-1) | 16,384 | Strictest |
| 02 high | 12-16GB | 2.0 | 2560 | 12,288 | High |
| 03 optimal | 10-12GB | 1.0 | 1920 | 8,192 | COLMAP defaults |
| 04 medium | 8-10GB | 2.5 | 1600 | 6,144 | Moderate |
| 05 low | 6-8GB | 2.0 | 1024 | 4,096 | Lenient |
| 06 lowest | 4-6GB | 1.0 | 1024 | 2,048 | Most lenient |
| 07 360 | — | 3.0 | 1024 | 8,192 | Standard |
| 08 360_2fps | — | 2.0 | 1024 | 8,192 | Standard |

**360 configs (07/08):** Specialized for equirectangular video. Convert each frame to 5 perspective views (front, right, back, left, top). Use sequential matching with vocab tree loop closure. Cap at 1000 total images.

## Usage

### Full pipeline (video to 3D model)

```bash
python script/fit_model_to_scene_full.py \
    --input <video.mp4> \
    --output_path <output_dir> \
    --config train_03_optimal_quality \
    --colmap_config colmap_03_optimal_quality
```

### Batch training with tmux

```bash
# Single video
./script/train_tmux.sh --input data/video.mp4 --output_path outputs/scene1 \
    --config train_03_optimal_quality --colmap_config colmap_03_optimal_quality

# Batch (all videos in a directory), 360 mode
./script/train_tmux.sh --batch --data_dir data/videos --output_base outputs/batch \
    --360 --config train_04_medium_quality --colmap_config colmap_08_360_2fps
```

Default configs for `train_tmux.sh`: `train_04_medium_quality` + `colmap_08_360_2fps`.

### Training only (COLMAP already done)

```bash
python script/train.py \
    train.gs_epochs=30000 \
    train.no_densify=True \
    gs.dataset.source_path=<colmap_scene> \
    gs.dataset.model_path=<output_dir>
```

## Tips

- **GPU memory during EDGS init:** The RoMa matcher is the peak memory consumer. Reduce `matches_per_ref`, `num_refs`, or `nns_per_ref` if you hit OOM during initialization.
- **GPU memory during training:** Reduce `batch_size`, `sh_degree`, or use `gs.dataset.data_device=cpu` to keep images on CPU.
- **COLMAP image size vs training image size:** COLMAP `max_image_size` controls reconstruction resolution. Training uses `gs.dataset.resolution` (default: -1 = original). These are independent.
- **360 video:** Always pair 360 COLMAP configs (07/08) with the `--360` flag in `fit_model_to_scene_full.py` or `train_tmux.sh`.
