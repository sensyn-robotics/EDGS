#!/usr/bin/env python
"""
GPU Memory Diagnostic Tool
Helps diagnose CUDA memory issues and suggests optimal settings
"""
import subprocess
import re

def get_gpu_info():
    """Get GPU memory information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
            for line in output.split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 4:
                    gpu_name, total_mem, free_mem, used_mem = parts
                    return {
                        'name': gpu_name,
                        'total': int(total_mem),
                        'free': int(free_mem),
                        'used': int(used_mem)
                    }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return None

def recommend_settings(gpu_info):
    """Recommend settings based on available GPU memory"""
    if not gpu_info:
        return
    
    free_mem = gpu_info['free']
    total_mem = gpu_info['total']
    
    print(f"\n🖥️  GPU: {gpu_info['name']}")
    print(f"💾 Total Memory: {total_mem} MB ({total_mem/1024:.1f} GB)")
    print(f"✅ Free Memory: {free_mem} MB ({free_mem/1024:.1f} GB)")
    print(f"🔧 Used Memory: {gpu_info['used']} MB ({gpu_info['used']/1024:.1f} GB)")
    
    print("\n📋 Recommended Settings:")
    
    # Conservative recommendations based on free memory
    if free_mem < 4000:
        print("⚠️  Very low free memory! Use minimal settings:")
        print("  --config train_very_low_memory")
        print("  --colmap_config very_low_memory")
        print("  --target_fps 1.0")
        print("  --max_image_size 800")
    elif free_mem < 6000:
        print("⚠️  Low free memory. Use conservative settings:")
        print("  --config train_very_low_memory")
        print("  --colmap_config very_low_memory")
        print("  --target_fps 2.0")
        print("  --max_image_size 1024")
    elif free_mem < 8000:
        print("⚡ Moderate free memory. Use low memory settings:")
        print("  --config train_low_memory")
        print("  --colmap_config low_memory")
        print("  --target_fps 2.0")
        print("  --max_image_size 1920")
    elif free_mem < 12000:
        print("✅ Good free memory. Use balanced settings:")
        print("  --config train_low_memory")
        print("  --colmap_config balanced")
        print("  --target_fps 3.0")
        print("  --max_image_size 1920")
    else:
        print("🚀 Excellent free memory! Use high quality settings:")
        print("  --config train_high_quality")
        print("  --colmap_config high_accuracy")
        print("  --target_fps 3.0")
        print("  --max_image_size -1")
    
    print("\n💡 Tips to free up GPU memory:")
    print("  1. Close other GPU applications (browsers, IDEs, etc.)")
    print("  2. Run: nvidia-smi to check what's using GPU memory")
    print("  3. In Docker, restart the container to clear memory")
    print("  4. Consider using smaller max_image_size or lower target_fps")

if __name__ == "__main__":
    gpu_info = get_gpu_info()
    if gpu_info:
        recommend_settings(gpu_info)
    else:
        print("❌ Could not get GPU information. Make sure nvidia-smi is available.")