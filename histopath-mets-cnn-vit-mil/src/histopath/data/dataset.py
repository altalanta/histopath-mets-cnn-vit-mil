"""Dataset classes for histopathology data."""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.seed import worker_init_fn


class HistopathDataset(Dataset):
    """Dataset for individual histopathology tiles."""
    
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        transform: Optional[Any] = None,
        image_col: str = "tile_path",
        label_col: str = "label",
        root_dir: Optional[Path] = None
    ):
        """
        Initialize histopathology dataset.
        
        Args:
            tiles_df: DataFrame with tile information
            transform: Albumentations transform to apply
            image_col: Column name for image paths
            label_col: Column name for labels
            root_dir: Root directory for image paths
        """
        self.tiles_df = tiles_df.reset_index(drop=True)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.root_dir = Path(root_dir) if root_dir else None
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.tiles_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        row = self.tiles_df.iloc[idx]
        
        # Load image
        image_path = row[self.image_col]
        if self.root_dir:
            image_path = self.root_dir / image_path
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get label
        label = row[label_col]
        
        # Create sample dictionary
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'tile_id': row.get('tile_id', f'tile_{idx}'),
            'slide_id': row.get('slide_id', 'unknown'),
        }
        
        # Add coordinates if available
        if 'x' in row and 'y' in row:
            sample['coords'] = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        
        return sample


class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning bags."""
    
    def __init__(
        self,
        bags_df: pd.DataFrame,
        tiles_df: Optional[pd.DataFrame] = None,
        transform: Optional[Any] = None,
        max_tiles_per_bag: Optional[int] = None,
        root_dir: Optional[Path] = None,
        preload_images: bool = False
    ):
        """
        Initialize MIL dataset.
        
        Args:
            bags_df: DataFrame with bag information
            tiles_df: DataFrame with tile information (optional)
            transform: Albumentations transform to apply
            max_tiles_per_bag: Maximum tiles per bag (for sampling)
            root_dir: Root directory for image paths
            preload_images: Whether to preload all images in memory
        """
        self.bags_df = bags_df.reset_index(drop=True)
        self.tiles_df = tiles_df
        self.transform = transform
        self.max_tiles_per_bag = max_tiles_per_bag
        self.root_dir = Path(root_dir) if root_dir else None
        self.preload_images = preload_images
        
        # Preload images if requested
        if self.preload_images:
            self._preload_all_images()
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.bags_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get bag by index."""
        bag_row = self.bags_df.iloc[idx]
        
        # Get tile information
        if self.tiles_df is not None:
            # Use tiles DataFrame for detailed tile info
            bag_tiles = self.tiles_df[
                self.tiles_df['slide_id'] == bag_row['slide_id']
            ].copy()
        else:
            # Parse tile info from bag row
            bag_tiles = self._parse_bag_tiles(bag_row)
        
        # Sample tiles if needed
        if self.max_tiles_per_bag and len(bag_tiles) > self.max_tiles_per_bag:
            bag_tiles = bag_tiles.sample(
                n=self.max_tiles_per_bag, 
                random_state=42
            ).reset_index(drop=True)
        
        # Load images
        images = []
        coords = []
        
        for _, tile_row in bag_tiles.iterrows():
            if self.preload_images:
                # Use preloaded image
                tile_id = tile_row.get('tile_id', f"tile_{tile_row.name}")
                image = self._preloaded_images[tile_id]
            else:
                # Load image on demand
                image = self._load_tile_image(tile_row)
            
            images.append(image)
            
            # Get coordinates
            if 'x' in tile_row and 'y' in tile_row:
                coords.append([tile_row['x'], tile_row['y']])
            else:
                coords.append([0, 0])
        
        # Stack images and coordinates
        images = torch.stack(images)
        coords = torch.tensor(coords, dtype=torch.float32)
        
        # Get bag label
        label = bag_row['label']
        
        # Create bag sample
        sample = {
            'images': images,  # Shape: (n_tiles, C, H, W)
            'coords': coords,  # Shape: (n_tiles, 2)
            'label': torch.tensor(label, dtype=torch.long),
            'bag_id': bag_row.get('bag_id', f'bag_{idx}'),
            'slide_id': bag_row['slide_id'],
            'n_tiles': len(images),
        }
        
        return sample
    
    def _parse_bag_tiles(self, bag_row: pd.Series) -> pd.DataFrame:
        """Parse tile information from bag row."""
        # Handle different formats of stored tile info
        tile_ids = bag_row.get('tile_ids', [])
        tile_paths = bag_row.get('tile_paths', [])
        coords = bag_row.get('coords', [])
        
        # Parse string representations if needed
        if isinstance(tile_ids, str):
            tile_ids = ast.literal_eval(tile_ids)
        if isinstance(tile_paths, str):
            tile_paths = ast.literal_eval(tile_paths)
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)
        
        # Create DataFrame
        tiles_data = []
        for i, tile_id in enumerate(tile_ids):
            tile_data = {
                'tile_id': tile_id,
                'slide_id': bag_row['slide_id'],
                'label': bag_row['label'],
            }
            
            if tile_paths and i < len(tile_paths):
                tile_data['tile_path'] = tile_paths[i]
            
            if coords and i < len(coords):
                if isinstance(coords[i], (list, tuple)) and len(coords[i]) >= 2:
                    tile_data['x'] = coords[i][0]
                    tile_data['y'] = coords[i][1]
            
            tiles_data.append(tile_data)
        
        return pd.DataFrame(tiles_data)
    
    def _load_tile_image(self, tile_row: pd.Series) -> torch.Tensor:
        """Load and transform a single tile image."""
        # Get image path
        if 'tile_path' in tile_row:
            image_path = tile_row['tile_path']
        else:
            # Construct path from tile_id
            tile_id = tile_row['tile_id']
            slide_id = tile_row['slide_id']
            image_path = f"images/{slide_id}/{tile_id}.png"
        
        if self.root_dir:
            image_path = self.root_dir / image_path
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image
    
    def _preload_all_images(self) -> None:
        """Preload all images in memory."""
        self._preloaded_images = {}
        
        for idx in range(len(self.bags_df)):
            bag_row = self.bags_df.iloc[idx]
            
            if self.tiles_df is not None:
                bag_tiles = self.tiles_df[
                    self.tiles_df['slide_id'] == bag_row['slide_id']
                ]
            else:
                bag_tiles = self._parse_bag_tiles(bag_row)
            
            for _, tile_row in bag_tiles.iterrows():
                tile_id = tile_row.get('tile_id', f"tile_{tile_row.name}")
                if tile_id not in self._preloaded_images:
                    self._preloaded_images[tile_id] = self._load_tile_image(tile_row)


class HDF5Dataset(Dataset):
    """Dataset for loading tiles from HDF5 files."""
    
    def __init__(
        self,
        hdf5_path: Path,
        transform: Optional[Any] = None,
        subset_indices: Optional[List[int]] = None
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            transform: Albumentations transform to apply
            subset_indices: Subset of indices to use
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        
        # Load metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['images'])
            self.labels = f['labels'][:]
            
            # Load other metadata if available
            self.tile_ids = [tid.decode() for tid in f['tile_ids'][:]] if 'tile_ids' in f else None
            self.coords = f['coords'][:] if 'coords' in f else None
        
        # Apply subset if specified
        if subset_indices is not None:
            self.indices = subset_indices
            self.length = len(subset_indices)
        else:
            self.indices = list(range(self.length))
    
    def __len__(self) -> int:
        """Get dataset length."""
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        # Map to actual index
        actual_idx = self.indices[idx]
        
        # Load image from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            image = f['images'][actual_idx]
            label = self.labels[actual_idx]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Create sample
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'index': actual_idx,
        }
        
        # Add metadata if available
        if self.tile_ids:
            sample['tile_id'] = self.tile_ids[actual_idx]
        
        if self.coords is not None:
            sample['coords'] = torch.tensor(
                self.coords[actual_idx], 
                dtype=torch.float32
            )
        
        return sample


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )