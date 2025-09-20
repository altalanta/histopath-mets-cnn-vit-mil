"""Utilities for tiling whole slide images."""

import math
from typing import Generator, List, Tuple

import numpy as np
import openslide
from PIL import Image


class WSITiler:
    """Whole slide image tiler for extracting patches."""
    
    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 0,
        magnification: int = 10,
        max_tiles_per_slide: int = 10000,
    ):
        """
        Initialize WSI tiler.
        
        Args:
            tile_size: Size of extracted tiles (square)
            overlap: Overlap between adjacent tiles in pixels
            magnification: Target magnification level (5, 10, 20, 40)
            max_tiles_per_slide: Maximum number of tiles to extract per slide
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.magnification = magnification
        self.max_tiles_per_slide = max_tiles_per_slide
        self.stride = tile_size - overlap
        
    def get_tile_coordinates(
        self, 
        slide_path: str,
        tissue_mask: np.ndarray = None
    ) -> List[Tuple[int, int, int]]:
        """
        Generate tile coordinates for a whole slide image.
        
        Args:
            slide_path: Path to the slide file
            tissue_mask: Optional tissue mask to filter coordinates
            
        Returns:
            List of (x, y, level) coordinates
        """
        slide = openslide.OpenSlide(slide_path)
        
        # Get appropriate level for target magnification
        level = self._get_magnification_level(slide)
        
        # Get dimensions at the target level
        level_dims = slide.level_dimensions[level]
        width, height = level_dims
        
        # Calculate number of tiles
        n_tiles_x = math.ceil((width - self.tile_size) / self.stride) + 1
        n_tiles_y = math.ceil((height - self.tile_size) / self.stride) + 1
        
        coordinates = []
        
        for y_idx in range(n_tiles_y):
            for x_idx in range(n_tiles_x):
                x = x_idx * self.stride
                y = y_idx * self.stride
                
                # Ensure tile doesn't go beyond image boundaries
                if x + self.tile_size > width:
                    x = width - self.tile_size
                if y + self.tile_size > height:
                    y = height - self.tile_size
                
                # Skip if coordinates are negative
                if x < 0 or y < 0:
                    continue
                
                # Check tissue mask if provided
                if tissue_mask is not None:
                    if not self._has_sufficient_tissue(tissue_mask, x, y, level, slide):
                        continue
                
                coordinates.append((x, y, level))
                
                # Limit number of tiles
                if len(coordinates) >= self.max_tiles_per_slide:
                    break
            
            if len(coordinates) >= self.max_tiles_per_slide:
                break
        
        slide.close()
        return coordinates
    
    def extract_tile(
        self, 
        slide_path: str, 
        x: int, 
        y: int, 
        level: int = 0
    ) -> Image.Image:
        """
        Extract a single tile from a slide.
        
        Args:
            slide_path: Path to the slide file
            x: X coordinate at the specified level
            y: Y coordinate at the specified level  
            level: Pyramid level to extract from
            
        Returns:
            PIL Image of the extracted tile
        """
        slide = openslide.OpenSlide(slide_path)
        
        try:
            # Convert coordinates to level 0 (highest resolution)
            downsample = slide.level_downsamples[level]
            x_level0 = int(x * downsample)
            y_level0 = int(y * downsample)
            
            # Extract tile
            tile = slide.read_region(
                (x_level0, y_level0), 
                level, 
                (self.tile_size, self.tile_size)
            )
            
            # Convert RGBA to RGB
            tile = tile.convert('RGB')
            
        finally:
            slide.close()
            
        return tile
    
    def extract_tiles_generator(
        self, 
        slide_path: str,
        coordinates: List[Tuple[int, int, int]]
    ) -> Generator[Tuple[Image.Image, int, int, int], None, None]:
        """
        Generator for extracting multiple tiles efficiently.
        
        Args:
            slide_path: Path to the slide file
            coordinates: List of (x, y, level) coordinates
            
        Yields:
            Tuple of (tile_image, x, y, level)
        """
        slide = openslide.OpenSlide(slide_path)
        
        try:
            for x, y, level in coordinates:
                # Convert coordinates to level 0
                downsample = slide.level_downsamples[level]
                x_level0 = int(x * downsample)
                y_level0 = int(y * downsample)
                
                # Extract tile
                tile = slide.read_region(
                    (x_level0, y_level0),
                    level,
                    (self.tile_size, self.tile_size)
                )
                
                # Convert RGBA to RGB
                tile = tile.convert('RGB')
                
                yield tile, x, y, level
                
        finally:
            slide.close()
    
    def _get_magnification_level(self, slide: openslide.OpenSlide) -> int:
        """
        Get the appropriate pyramid level for target magnification.
        
        Args:
            slide: OpenSlide object
            
        Returns:
            Pyramid level index
        """
        # Get objective power (base magnification)
        try:
            objective_power = float(slide.properties.get(
                openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40
            ))
        except (ValueError, TypeError):
            objective_power = 40.0  # Default assumption
        
        # Calculate target downsample factor
        target_downsample = objective_power / self.magnification
        
        # Find closest level
        level_downsamples = slide.level_downsamples
        best_level = 0
        min_diff = float('inf')
        
        for level, downsample in enumerate(level_downsamples):
            diff = abs(downsample - target_downsample)
            if diff < min_diff:
                min_diff = diff
                best_level = level
        
        return best_level
    
    def _has_sufficient_tissue(
        self, 
        tissue_mask: np.ndarray, 
        x: int, 
        y: int, 
        level: int,
        slide: openslide.OpenSlide,
        threshold: float = 0.3
    ) -> bool:
        """
        Check if a tile location has sufficient tissue content.
        
        Args:
            tissue_mask: Binary tissue mask
            x: X coordinate
            y: Y coordinate
            level: Pyramid level
            slide: OpenSlide object
            threshold: Minimum tissue ratio required
            
        Returns:
            True if tile has sufficient tissue
        """
        # Calculate scaling factor between mask and tile coordinates
        mask_height, mask_width = tissue_mask.shape
        level_width, level_height = slide.level_dimensions[level]
        
        scale_x = mask_width / level_width
        scale_y = mask_height / level_height
        
        # Map tile coordinates to mask coordinates
        mask_x = int(x * scale_x)
        mask_y = int(y * scale_y)
        mask_tile_size = int(self.tile_size * min(scale_x, scale_y))
        
        # Extract mask region
        mask_x_end = min(mask_x + mask_tile_size, mask_width)
        mask_y_end = min(mask_y + mask_tile_size, mask_height)
        
        if mask_x >= mask_width or mask_y >= mask_height:
            return False
        
        mask_region = tissue_mask[mask_y:mask_y_end, mask_x:mask_x_end]
        
        if mask_region.size == 0:
            return False
        
        tissue_ratio = np.sum(mask_region) / mask_region.size
        return tissue_ratio >= threshold