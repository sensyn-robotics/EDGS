# EDGS Training Configuration Guide

This directory contains various training configurations for EDGS (Eliminating Densification for Gaussian Splatting), optimized for different GPU memory constraints and quality requirements.

## Configuration Hierarchy

The configurations are ranked from **highest to lowest quality/accuracy**:

### 1. **train_01_highest_quality.yaml** - Highest Quality ⭐⭐⭐⭐⭐
- **Target GPU**: 16GB+ VRAM (RTX 4090, RTX 3090, A100, etc.)
- **Quality**: Absolute best possible reconstruction quality
- **Settings**:
  - 60,000 training epochs
  - Batch size: 64
  - SH degree: 3 (full spherical harmonics)
  - Full EDGS correlation initialization
  - High learning rates for optimal convergence
- **Memory Usage**: High (~16GB+)
- **Training Time**: ~3-4 hours
- **Expected PSNR**: 25-35+
- **Use when**: Maximum quality is needed and you have high-end hardware

### 2. **train_02_high_quality.yaml** - High Quality ⭐⭐⭐⭐
- **Target GPU**: 12-16GB VRAM (RTX 4080, RTX 3080 Ti, etc.)
- **Quality**: Extended training with progressive densification
- **Settings**:
  - 100,000 training epochs (very long training)
  - Batch size: 1
  - SH degree: 1 (memory efficient)
  - Progressive densification strategy
  - No EDGS correlation (memory conservative)
- **Memory Usage**: Medium-high (~12-16GB)
- **Training Time**: ~3-4 hours
- **Expected PSNR**: 18-25
- **Use when**: You want excellent quality with 12-16GB GPU and can wait

### 3. **train_03_optimal_quality.yaml** - Optimal Quality ⭐⭐⭐
- **Target GPU**: 12-16GB VRAM (RTX 4080, RTX 3080 Ti, etc.)
- **Quality**: Best compromise between quality and training time
- **Settings**:
  - 30,000 training epochs
  - Batch size: 12
  - SH degree: 3 (full spherical harmonics)
  - Full EDGS correlation initialization
  - Optimized learning rates
- **Memory Usage**: Medium-high (~12-16GB)
- **Training Time**: ~1.5-2 hours
- **Expected PSNR**: 22-28
- **Use when**: You want high quality with reasonable training time

### 4. **train_04_medium_quality.yaml** - Medium Quality ⭐⭐⭐
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
- **Expected PSNR**: 16-20
- **Use when**: You need good quality but have limited GPU memory

### 5. **train_05_low_quality.yaml** - Low Quality ⭐⭐
- **Target GPU**: 8-12GB VRAM (RTX 4060, RTX 3060 Ti, RTX 4050, etc.)
- **Quality**: Acceptable quality with memory-safe settings
- **Settings**:
  - 80,000 training epochs (longer training compensates)
  - Batch size: 1
  - SH degree: 1 (reduced spherical harmonics)
  - Controlled densification
  - SfM-only initialization (no EDGS correlation)
- **Memory Usage**: Low (~8-12GB)
- **Training Time**: ~2.5-3 hours
- **Expected PSNR**: 15-18
- **Use when**: You have memory constraints but can afford longer training time

### 6. **train_06_lowest_quality.yaml** - Lowest Quality ⭐
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
- **Expected PSNR**: 12-15
- **Use when**: You have severe memory limitations or need quick results

## Usage Examples

### Running with Different Configurations

```bash
# Highest quality (requires 16GB+ GPU) - Best possible results
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_highest \
    --config train_01_highest_quality \
    --max_image_size 1920

# High quality (requires 12-16GB GPU) - Extended training
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_high \
    --config train_02_high_quality \
    --max_image_size 800

# Optimal quality (requires 12-16GB GPU) - Balanced approach
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_optimal \
    --config train_03_optimal_quality \
    --max_image_size 1600

# Medium quality (8-12GB GPU) - Good compromise
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_medium \
    --config train_04_medium_quality \
    --max_image_size 1024

# Low quality (8-12GB GPU) - Memory safe, long training
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_low \
    --config train_05_low_quality \
    --max_image_size 512

# Lowest quality (4-8GB GPU) - Emergency settings
python script/fit_model_to_scene_full.py \
    --input <video.mp4 or image_dir> \
    --output_path outputs/my_scene_lowest \
    --config train_06_lowest_quality \
    --max_image_size 256
```

### Docker Script Usage

Convenient Docker scripts are available in the `script/` directory:

```bash
# Use the appropriate script for your hardware and quality needs
./script/run_docker_01_highest_quality.sh    # 16GB+ GPU - Best possible quality ⭐⭐⭐⭐⭐
./script/run_docker_02_high_quality.sh       # 12-16GB GPU - Extended training ⭐⭐⭐⭐  
./script/run_docker_03_optimal_quality.sh    # 12-16GB GPU - Balanced approach ⭐⭐⭐
./script/run_docker_04_medium_quality.sh     # 8-12GB GPU - Good compromise ⭐⭐⭐
./script/run_docker_05_low_quality.sh        # 8-12GB GPU - Memory safe ⭐⭐
./script/run_docker_06_lowest_quality.sh     # 4-8GB GPU - Emergency settings ⭐
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

| GPU Model | VRAM | Recommended Config | Docker Script | Max Image Size |
|-----------|------|--------------------|---------------|----------------|
| RTX 4090 | 24GB | train_01_highest_quality | run_docker_01_highest_quality.sh | 1920+ |
| RTX 4080 | 16GB | train_02_high_quality | run_docker_02_high_quality.sh | 800 |
| RTX 4080 | 12GB | train_03_optimal_quality | run_docker_03_optimal_quality.sh | 1600 |
| RTX 4070 Ti | 12GB | train_04_medium_quality | run_docker_04_medium_quality.sh | 1024 |
| RTX 4070 | 12GB | train_05_low_quality | run_docker_05_low_quality.sh | 800 |
| RTX 4060 Ti | 16GB | train_03_optimal_quality | run_docker_03_optimal_quality.sh | 1024 |
| RTX 4060 Ti | 8GB | train_05_low_quality | run_docker_05_low_quality.sh | 512 |
| RTX 4060 | 8GB | train_06_lowest_quality | run_docker_06_lowest_quality.sh | 256 |
| RTX 3080 | 10GB | train_05_low_quality | run_docker_05_low_quality.sh | 512 |
| RTX 3070 | 8GB | train_06_lowest_quality | run_docker_06_lowest_quality.sh | 256 |

## Quick Reference

### For Immediate Use:
```bash
# Best quality (if you have RTX 4090/A100)
./script/run_docker_01_highest_quality.sh

# Long training approach (RTX 4080, 3080 Ti)  
./script/run_docker_02_high_quality.sh

# Balanced (RTX 4080, 3080 Ti)
./script/run_docker_03_optimal_quality.sh

# Conservative (RTX 4070, 3070, 4060 Ti)
./script/run_docker_04_medium_quality.sh

# Memory safe (RTX 4060, 3060 Ti)
./script/run_docker_05_low_quality.sh

# Emergency (RTX 3050, mobile GPUs)
./script/run_docker_06_lowest_quality.sh
```

### Quality Expectations:
- **⭐⭐⭐⭐⭐ Highest**: PSNR 25-35+ (publication quality)
- **⭐⭐⭐⭐ High**: PSNR 18-25 (excellent quality)  
- **⭐⭐⭐ Optimal/Medium**: PSNR 16-22 (very good quality)
- **⭐⭐ Low**: PSNR 15-18 (acceptable quality)
- **⭐ Lowest**: PSNR 12-15 (basic reconstruction)

Choose the configuration that matches your hardware capabilities and quality requirements!