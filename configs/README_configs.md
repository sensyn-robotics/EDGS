# EDGS Configuration System

This directory contains configuration files for both EDGS training and COLMAP reconstruction.

## Training Configurations

Training configs control EDGS model optimization, batch sizes, learning rates, and initialization settings.

### Available Training Configs

1. **`train.yaml`** - Standard configuration
   - Default balanced settings
   - Suitable for most use cases
   - GPU Memory: 8-12GB recommended

2. **`train_high_quality.yaml`** - High quality mode
   - Best reconstruction quality
   - Larger batch sizes (64)
   - More iterations (60k)
   - Dense initialization (40k matches per ref, 500 refs)
   - GPU Memory: 12GB+ required

3. **`train_low_memory.yaml`** - Low memory mode
   - Good quality with reduced memory usage
   - Smaller batch size (16)
   - Moderate iterations (30k)
   - Balanced initialization (15k matches per ref, 180 refs)
   - GPU Memory: 6-8GB

4. **`train_very_low_memory.yaml`** - Very low memory mode
   - Minimal memory footprint
   - Minimal batch size (8)
   - Fewer iterations (20k)
   - Reduced initialization (10k matches per ref, 90 refs)
   - GPU Memory: 4-6GB

### Usage
```bash
python script/fit_model_to_scene_full.py --video_path <video> --config train_low_memory
```

## COLMAP Configurations

COLMAP configs control 3D reconstruction quality vs memory trade-offs during the initial camera pose estimation.

### Available COLMAP Configs

1. **`colmap_high_accuracy.yaml`** - Best quality
   - 16,384 SIFT features per image
   - 3200px max image size
   - Super-resolution features (first_octave: -1)
   - Memory: 16GB+ recommended

2. **`colmap_balanced.yaml`** - Good quality
   - 8,192 SIFT features per image
   - 2048px max image size
   - Standard feature extraction
   - Memory: 8-12GB

3. **`colmap_low_memory.yaml`** - Reduced memory (default)
   - 4,096 SIFT features per image
   - 1920px max image size
   - No super-resolution features
   - Memory: 6-8GB

4. **`colmap_very_low_memory.yaml`** - Minimal memory
   - 2,048 SIFT features per image
   - 1024px max image size
   - Minimal feature extraction
   - Memory: 4-6GB

### Usage
```bash
python script/fit_model_to_scene_full.py --video_path <video> --colmap_config very_low_memory
```

## Common Use Cases

### High-end System (16GB+ GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_high_quality \
    --colmap_config high_accuracy
```

### Standard System (8-12GB GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train \
    --colmap_config balanced
```

### Memory-Constrained System (6-8GB GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_low_memory \
    --colmap_config low_memory
```

### Minimal System (4-6GB GPU) or "Killed" Errors
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_very_low_memory \
    --colmap_config very_low_memory
```

## Key Differences Summary

| Config Level | Batch Size | SIFT Features | Image Size | GPU Memory |
|-------------|------------|---------------|------------|------------|
| High Quality | 64 | 16,384 | 3200px | 12-16GB+ |
| Standard | 32 | 8,192 | 2048px | 8-12GB |
| Low Memory | 16 | 4,096 | 1920px | 6-8GB |
| Very Low Memory | 8 | 2,048 | 1024px | 4-6GB |