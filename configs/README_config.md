# EDGS Training Configuration Guide

This directory contains various training configurations for EDGS (Eliminating Densification for Gaussian Splatting), optimized for different GPU memory constraints and quality requirements.

## Configuration Hierarchy

The configurations are ranked from **highest to lowest quality/accuracy**:

### 1. **train_high_quality.yaml** - Highest Quality ⭐⭐⭐⭐⭐
- **Target GPU**: 16GB+ VRAM (RTX 4090, RTX 3090, A100, etc.)
- **Quality**: Best possible reconstruction quality
- **Settings**:
  - 60,000 training epochs
  - Batch size: 64
  - SH degree: 3 (full spherical harmonics)
  - Full EDGS correlation initialization
  - High learning rates for optimal convergence
- **Memory Usage**: High (~16GB+)
- **Training Time**: ~3-4 hours
- **Use when**: Maximum quality is needed and you have high-end hardware

### 2. **train_optimal.yaml** - High Quality ⭐⭐⭐⭐
- **Target GPU**: 12-16GB VRAM (RTX 4080 Ti, RTX 3080 Ti, etc.)
- **Quality**: Excellent quality with balanced performance
- **Settings**:
  - 30,000 training epochs
  - Batch size: 12
  - SH degree: 3 (full spherical harmonics)
  - Full EDGS correlation initialization
  - Optimized learning rates
- **Memory Usage**: Medium-high (~12-16GB)
- **Training Time**: ~1.5-2 hours
- **Use when**: You want high quality with reasonable training time

### 3. **train_medium_quality.yaml** - Medium Quality ⭐⭐⭐
- **Target GPU**: 8-12GB VRAM (RTX 4070, RTX 3070, RTX 4060 Ti, etc.)
- **Quality**: Good quality with memory efficiency
- **Settings**:
  - 20,000 training epochs
  - Batch size: 1
  - SH degree: 1 (reduced spherical harmonics)
  - Limited densification enabled
  - EDGS correlation with reduced parameters
- **Memory Usage**: Medium (~8-12GB)
- **Training Time**: ~1-1.5 hours
- **Use when**: You need good quality but have limited GPU memory

### 4. **train_low_memory.yaml** - Lower Quality ⭐⭐
- **Target GPU**: 8-12GB VRAM (RTX 4060, RTX 3060 Ti, RTX 4050, etc.)
- **Quality**: Acceptable quality with memory-safe settings
- **Settings**:
  - 60,000 training epochs (longer training compensates for simpler model)
  - Batch size: 1
  - SH degree: 1 (reduced spherical harmonics)
  - No densification (EDGS principle only)
  - SfM-only initialization (no EDGS correlation)
- **Memory Usage**: Low (~8-12GB)
- **Training Time**: ~2-2.5 hours
- **Use when**: You have memory constraints but can afford longer training time

### 5. **train_ultra_low_memory.yaml** - Lowest Quality ⭐
- **Target GPU**: 4-8GB VRAM (RTX 3050, GTX 1660, mobile GPUs, etc.)
- **Quality**: Basic reconstruction quality
- **Settings**:
  - Minimal training epochs
  - Batch size: 1
  - SH degree: 0 (DC component only)
  - No densification
  - Minimal feature matching
  - Aggressive memory optimizations
- **Memory Usage**: Very low (~4-8GB)
- **Training Time**: ~30-60 minutes
- **Use when**: You have severe memory limitations or need quick results

## Usage Examples

### Running with Different Configurations

```bash
# High quality (requires 16GB+ GPU)
python script/fit_model_to_scene_full.py \
    --colmap_output_path outputs/my_scene \
    --output_path outputs/my_scene_high_quality \
    --config train_high_quality \
    --max_image_size 1920

# Optimal quality (requires 12-16GB GPU)  
python script/fit_model_to_scene_full.py \
    --colmap_output_path outputs/my_scene \
    --output_path outputs/my_scene_optimal \
    --config train_optimal \
    --max_image_size 1600

# Medium quality (8-12GB GPU)
python script/fit_model_to_scene_full.py \
    --colmap_output_path outputs/my_scene \
    --output_path outputs/my_scene_medium \
    --config train_medium_quality \
    --max_image_size 1024

# Low memory (8-12GB GPU, conservative)
python script/fit_model_to_scene_full.py \
    --colmap_output_path outputs/my_scene \
    --output_path outputs/my_scene_low_memory \
    --config train_low_memory \
    --max_image_size 512

# Ultra low memory (4-8GB GPU)
python script/fit_model_to_scene_full.py \
    --colmap_output_path outputs/my_scene \
    --output_path outputs/my_scene_ultra_low \
    --config train_ultra_low_memory \
    --max_image_size 256
```

### Docker Script Usage

Convenient Docker scripts are available in the `script/` directory:

```bash
# Use the appropriate script for your hardware
./script/run_docker_high_quality.sh      # 16GB+ GPU
./script/run_docker_optimal.sh           # 12-16GB GPU  
./script/run_docker_low_memory.sh        # 8-12GB GPU (conservative)
./script/run_docker_ultra_low_memory.sh  # 4-8GB GPU
```

## Configuration Parameters Explained

### Key Parameters That Affect Quality vs Memory:

- **gs_epochs**: More epochs = better quality but longer training time
- **batch_size**: Larger batches = better convergence but more memory
- **sh_degree**: Higher degree = better lighting/color but more memory
- **max_image_size**: Larger images = better detail but much more memory
- **EDGS correlation (init_wC.use)**: Better initialization but requires more memory
- **Densification**: Can improve quality but uses more memory

### Memory vs Quality Trade-offs:

- **SH Degree**: 0 → 1 → 3 (memory increases ~4x each step)
- **Batch Size**: 1 → 12 → 64 (memory scales linearly)  
- **Image Resolution**: 256 → 512 → 1024 → 1920 (memory scales quadratically)
- **EDGS Correlation**: Disabled → Enabled (adds ~2-4GB memory usage)

## Troubleshooting

### CUDA Out of Memory Errors:
1. Use a lower quality configuration
2. Reduce `max_image_size` parameter
3. Close other GPU applications
4. Restart Docker container to clear GPU memory

### Poor Quality Results:
1. Use a higher quality configuration if you have enough GPU memory
2. Increase training epochs
3. Use larger input images (if memory allows)
4. Enable EDGS correlation initialization

### Expected Quality Ranges:
- **High Quality**: PSNR > 18, SSIM > 0.65
- **Medium Quality**: PSNR > 16, SSIM > 0.55  
- **Low Memory**: PSNR > 15, SSIM > 0.52
- **Ultra Low**: PSNR > 12, SSIM > 0.45

## Hardware Recommendations

| GPU Model | VRAM | Recommended Config | Max Image Size |
|-----------|------|-------------------|----------------|
| RTX 4090 | 24GB | train_high_quality | 2048+ |
| RTX 4080 | 16GB | train_optimal | 1600 |
| RTX 4070 Ti | 12GB | train_medium_quality | 1024 |
| RTX 4070 | 12GB | train_low_memory | 800 |
| RTX 4060 Ti | 16GB | train_optimal | 1024 |
| RTX 4060 Ti | 8GB | train_low_memory | 512 |
| RTX 4060 | 8GB | train_ultra_low_memory | 256 |
| RTX 3080 | 10GB | train_low_memory | 512 |
| RTX 3070 | 8GB | train_ultra_low_memory | 256 |

Choose the configuration that matches your hardware capabilities and quality requirements!