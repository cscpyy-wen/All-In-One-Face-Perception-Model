#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing and mathematical utility functions.

Provides all low-level image I/O, normalization, colorization, and
numerical helpers used by the inference pipeline and visualizer.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================================
# Normalization constants
# ============================================================================

# Classification tasks (recog / age / biattr / affect)
CLASSIFICATION_MEAN = [0.5, 0.5, 0.5]
CLASSIFICATION_STD = [0.5, 0.5, 0.5]

# Dense prediction tasks (parsing / align) — FaRL / CLIP statistics
DENSE_MEAN = [0.48145466, 0.4578275, 0.40821073]
DENSE_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================================
# Image I/O
# ============================================================================

def read_image_rgb(image_path: str | np.ndarray) -> np.ndarray:
    """Read an image from disk and return it as an RGB numpy array."""
    image = Image.open(str(image_path)).convert("RGB")
    return np.asarray(image)


def save_rgb(path, image_rgb: np.ndarray) -> None:
    """Save an RGB numpy array as an image file."""
    Image.fromarray(image_rgb.astype(np.uint8)).save(str(path))


# ============================================================================
# Geometric transforms
# ============================================================================

def resize_to_hw(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to exact (width, height)."""
    img = Image.fromarray(image_rgb)
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img)


def resize_rgb(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an image to (size, size), preserving square aspect ratio."""
    return resize_to_hw(image_rgb, size, size)


# ============================================================================
# Tensor operations
# ============================================================================

def normalize(image_rgb: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    """Convert an RGB uint8 image to a normalized float32 tensor (C, H, W)."""
    x = image_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = (x - mean_t) / std_t
    return x


# ============================================================================
# Numerical helpers
# ============================================================================

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid for numpy arrays."""
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# Parsing visualization
# ============================================================================

# Fixed color map for 19 parsing classes
PARSING_COLORS = np.array([
    [  0,   0,   0],   # 0  background
    [255, 140, 200],   # 1  neck  — pink
    [  0, 200, 255],   # 2  face  — cyan
    [180, 130, 100],   # 3  cloth
    [  0, 255,   0],   # 4  rr
    [  0, 255,   0],   # 5  lr
    [  0, 255,   0],   # 6  rb
    [  0, 255,   0],   # 7  lb
    [255, 255,   0],   # 8  re
    [255, 255,   0],   # 9  le
    [255, 200,   0],   # 10 nose
    [200,  80,  80],   # 11 imouth
    [200,  80,  80],   # 12 llip
    [200,  80,  80],   # 13 ulip
    [  0, 180,   0],   # 14 hair
    [  0, 128, 255],   # 15 glass
    [128,   0, 255],   # 16 hat
    [255, 128,   0],   # 17 earr
    [128, 128, 255],   # 18 neckl
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Map an integer mask (H, W) to an RGB image using PARSING_COLORS."""
    colors = PARSING_COLORS[:num_classes]
    return colors[mask]


def overlay_mask(
    image_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend a colored mask onto an RGB image."""
    img = image_rgb.astype(np.float32)
    msk = mask_rgb.astype(np.float32)
    out = img * (1 - alpha) + msk * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================================
# Landmark drawing
# ============================================================================

def draw_landmarks(
    image_rgb: np.ndarray,
    landmarks: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Draw green circles at each landmark point on a copy of the image."""
    img = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for x, y in landmarks.astype(np.float32):
        left_up = (float(x) - radius, float(y) - radius)
        right_down = (float(x) + radius, float(y) + radius)
        draw.ellipse([left_up, right_down], fill=(0, 255, 0), outline=(0, 255, 0))

    return np.asarray(img)
