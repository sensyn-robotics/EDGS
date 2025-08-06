# EDGS Configuration System

This directory contains configuration files for both EDGS training and COLMAP reconstruction.

## Training Configurations

Training configs control EDGS model optimization, batch sizes, learning rates, and initialization settings.

### Available Training Configs

| Config | GPU Memory | Batch Size | Initialization | Use Case |
|--------|------------|------------|----------------|----------|
| **`train.yaml`** | 8-12GB | 32 | EDGS (RoMa) | Default balanced settings |
| **`train_high_quality.yaml`** | 16GB+ | 64 | EDGS + SfM | Best quality, long training (60k iters) |
| **`train_optimal.yaml`** | 12GB+ | 12 | EDGS (RoMa) | Good quality/speed balance (30k iters) |
| **`train_low_memory.yaml`** | 8-12GB | 8 | SfM-only* | Avoids RoMa OOM issues (25k iters) |
| **`train_ultra_low_memory.yaml`** | <8GB | 4 | SfM-only* | Emergency fallback (15k iters) |

*SfM-only: Uses only COLMAP points instead of EDGS correlation initialization to avoid loading the RoMa model which can cause CUDA OOM.

### Key Differences:
- **EDGS initialization (`init_wC.use: True`)**: Uses RoMa model for dense correspondence matching
  - Better initialization quality
  - Requires ~4GB additional GPU memory during initialization
  - Can cause CUDA OOM on loading the model
  
- **SfM-only (`init_wC.use: False, add_SfM_init: True`)**: Uses only COLMAP sparse points
  - Lower memory usage (no RoMa model)
  - Slightly reduced initialization quality
  - Reliable fallback for memory-constrained systems

### Docker Scripts

For easy usage with proper memory management:

```bash
# Optimal performance (12GB+ GPU)
./script/run_docker_optimal.sh  # uses train_optimal.yaml

# Low memory mode (8-12GB GPU) 
./script/run_docker_low_memory.sh  # uses train_low_memory.yaml

# Ultra low memory (<8GB GPU)
./script/run_docker_ultra_low_memory.sh  # uses train_ultra_low_memory.yaml
```

## COLMAP Configurations

COLMAP configs control 3D reconstruction quality vs memory trade-offs during the initial camera pose estimation.

### Available COLMAP Configs

| Config | Max Image Size | SIFT Features | GPU Memory | Quality |
|--------|----------------|---------------|------------|---------|
| **`colmap_high_accuracy.yaml`** | 3200px | 16,384 | 16GB+ | Best |
| **`colmap_balanced.yaml`** | 2048px | 8,192 | 8-12GB | Good |
| **`colmap_low_memory.yaml`** | 1920px | 4,096 | 6-8GB | Fair |
| **`colmap_very_low_memory.yaml`** | 1024px | 2,048 | 4-6GB | Basic |

## Common Use Cases

### High-end System (16GB+ GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_high_quality \
    --colmap_config colmap_high_accuracy
```

### Standard System (12GB GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_optimal \
    --colmap_config colmap_balanced
```

### Memory-Constrained System (8-12GB GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_low_memory \
    --colmap_config colmap_low_memory
```

### Minimal System (<8GB GPU)
```bash
python script/fit_model_to_scene_full.py \
    --video_path data/video.mp4 \
    --config train_ultra_low_memory \
    --colmap_config colmap_very_low_memory \
    --max_image_size 800  # Further reduce if needed
```

## Memory Troubleshooting

### CUDA Out of Memory Errors

1. **During initialization** (loading RoMa model):
   - Switch to a config with SfM-only initialization (`train_low_memory.yaml` or `train_ultra_low_memory.yaml`)

2. **During training**:
   - Reduce batch size in the config
   - Use a lower memory config
   - Reduce `--max_image_size` parameter

3. **During COLMAP**:
   - Use a lower memory COLMAP config
   - Reduce `--max_image_size` parameter

### Docker-Specific Tips

The Docker scripts in `script/run_docker_*.sh` include:
- Automatic memory cleanup before/after training
- PyTorch memory optimization settings
- Proper conda environment activation
- Helpful error messages for troubleshooting

## Creating Custom Configs

To create your own configuration:

1. Copy an existing config as template
2. Modify parameters as needed
3. Use `defaults` section to inherit settings:
   ```yaml
   defaults:
     - train  # inherits from train.yaml
     - _self_
   ```

Key parameters to adjust for memory:
- `gs.opt.batch_size`: Smaller = less memory
- `init_wC.matches_per_ref`: Fewer = less memory
- `init_wC.num_refs`: Fewer = less memory
- `init_wC.use`: False = avoid RoMa model loading