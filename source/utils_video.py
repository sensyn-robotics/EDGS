#!/usr/bin/env python
# coding: utf-8

"""
Video processing utilities for EDGS.

This module re-exports video processing functions for backwards compatibility.
The actual implementations are in separate files:
- convert_equirectangular_to_cubemap.py
- process_video_to_colmap_scene.py
"""

# Re-export functions from their individual modules
from source.convert_equirectangular_to_cubemap import convert_equirectangular_to_cubemap
from source.process_video_to_colmap_scene import (
    process_video_to_colmap_scene,
    check_colmap_scene,
    find_videos_in_directory,
    find_images_in_directory,
)

__all__ = [
    'convert_equirectangular_to_cubemap',
    'process_video_to_colmap_scene',
    'check_colmap_scene',
    'find_videos_in_directory',
    'find_images_in_directory',
]
