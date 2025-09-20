"""Tissue detection and masking utilities for whole slide images."""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def create_tissue_mask(
    image: np.ndarray,
    method: str = "otsu",
    threshold: float = 0.8,
    min_tissue_ratio: float = 0.1,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Create a binary tissue mask from an RGB image.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        method: Method for tissue detection ('otsu', 'simple_threshold', 'morphology')
        threshold: Threshold value for simple thresholding (0-1)
        min_tissue_ratio: Minimum tissue content ratio to consider valid
        kernel_size: Kernel size for morphological operations
        
    Returns:
        Binary tissue mask (H, W) with 1 for tissue, 0 for background
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")
    
    # Convert to different color spaces for better tissue detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    if method == "otsu":
        # Use saturation channel from HSV for Otsu thresholding
        saturation = hsv[:, :, 1]
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif method == "simple_threshold":
        # Simple thresholding on saturation
        saturation = hsv[:, :, 1]
        threshold_val = int(threshold * 255)
        _, mask = cv2.threshold(saturation, threshold_val, 255, cv2.THRESH_BINARY)
        
    elif method == "morphology":
        # Morphological operations on L channel from LAB
        l_channel = lab[:, :, 0]
        
        # Initial threshold
        _, mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    else:
        raise ValueError(f"Unknown tissue detection method: {method}")
    
    # Convert to binary (0, 1)
    mask = (mask > 0).astype(np.uint8)
    
    # Filter out regions with too little tissue
    tissue_ratio = np.sum(mask) / mask.size
    if tissue_ratio < min_tissue_ratio:
        mask = np.zeros_like(mask)
    
    return mask


def filter_tile_by_tissue_content(
    tile: np.ndarray,
    tissue_threshold: float = 0.3,
    **mask_kwargs
) -> bool:
    """
    Determine if a tile contains sufficient tissue content.
    
    Args:
        tile: RGB tile image (H, W, 3)
        tissue_threshold: Minimum fraction of tissue required
        **mask_kwargs: Additional arguments for create_tissue_mask
        
    Returns:
        True if tile has sufficient tissue content
    """
    mask = create_tissue_mask(tile, **mask_kwargs)
    tissue_ratio = np.sum(mask) / mask.size
    return tissue_ratio >= tissue_threshold


def is_background_tile(tile: np.ndarray, background_threshold: float = 0.85) -> bool:
    """
    Check if a tile is mostly background (white/empty).
    
    Args:
        tile: RGB tile image (H, W, 3)
        background_threshold: Threshold for background detection (0-1)
        
    Returns:
        True if tile is mostly background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    
    # Calculate percentage of pixels above threshold (white/light)
    white_pixels = np.sum(gray > 200)  # Assuming 200+ is background
    total_pixels = gray.size
    
    background_ratio = white_pixels / total_pixels
    return background_ratio >= background_threshold


def remove_small_objects(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small connected components from a binary mask.
    
    Args:
        mask: Binary mask (H, W)
        min_size: Minimum size of objects to keep
        
    Returns:
        Cleaned binary mask
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # Create output mask
    cleaned_mask = np.zeros_like(mask)
    
    # Keep only large enough components
    for label in range(1, num_labels):
        component_mask = (labels == label)
        if np.sum(component_mask) >= min_size:
            cleaned_mask[component_mask] = 1
    
    return cleaned_mask


def get_tissue_bbox(mask: np.ndarray, padding: int = 0) -> Tuple[int, int, int, int]:
    """
    Get bounding box of tissue regions in a mask.
    
    Args:
        mask: Binary tissue mask (H, W)
        padding: Padding to add around bounding box
        
    Returns:
        Bounding box as (x_min, y_min, x_max, y_max)
    """
    # Find coordinates of tissue pixels
    tissue_coords = np.where(mask > 0)
    
    if len(tissue_coords[0]) == 0:
        # No tissue found
        return (0, 0, mask.shape[1], mask.shape[0])
    
    y_min, y_max = tissue_coords[0].min(), tissue_coords[0].max()
    x_min, x_max = tissue_coords[1].min(), tissue_coords[1].max()
    
    # Add padding
    h, w = mask.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)