"""Data modules for PyTorch Lightning."""

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..data.dataset import HistopathDataset, MILDataset, create_dataloader
from ..data.transforms import HistopathTransforms


class HistopathDataModule(pl.LightningDataModule):
    """Data module for histopathology tile classification."""
    
    def __init__(
        self,
        data_dir: Path,
        tiles_index: str = "tiles_index.parquet",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_prob: float = 0.5,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize histopathology data module.
        
        Args:
            data_dir: Directory containing processed data
            tiles_index: Name of tiles index file
            image_size: Input image size
            batch_size: Batch size for training
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            augment_prob: Probability for augmentations
            normalize: Whether to normalize images
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.tiles_index = tiles_index
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_prob = augment_prob
        self.normalize = normalize
        
        # Will be set in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Transforms
        self.train_transform = HistopathTransforms.get_train_transforms(
            size=image_size,
            normalize=normalize,
            augment_prob=augment_prob
        )
        self.val_transform = HistopathTransforms.get_val_transforms(
            size=image_size,
            normalize=normalize
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        
        # Load tiles index
        tiles_path = self.data_dir / self.tiles_index
        if tiles_path.suffix == '.parquet':
            tiles_df = pd.read_parquet(tiles_path)
        elif tiles_path.suffix == '.csv':
            tiles_df = pd.read_csv(tiles_path)
        else:
            raise ValueError(f"Unsupported tiles index format: {tiles_path.suffix}")
        
        # Load slide metadata for splits
        metadata_path = self.data_dir.parent / "processed" / "slide_metadata.csv"
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            # Merge to get split information
            tiles_df = tiles_df.merge(
                metadata_df[['slide_id', 'split']], 
                on='slide_id', 
                how='left'
            )
        
        if stage == "fit" or stage is None:
            # Training data
            if 'split' in tiles_df.columns:
                train_tiles = tiles_df[tiles_df['split'] == 'train']
                val_tiles = tiles_df[tiles_df['split'] == 'val']
            else:
                # Fallback: use all data for training, empty validation
                train_tiles = tiles_df
                val_tiles = tiles_df.iloc[:0]  # Empty DataFrame
            
            self.train_dataset = HistopathDataset(
                tiles_df=train_tiles,
                transform=self.train_transform,
                root_dir=self.data_dir
            )
            
            self.val_dataset = HistopathDataset(
                tiles_df=val_tiles,
                transform=self.val_transform,
                root_dir=self.data_dir
            )
        
        if stage == "test" or stage is None:
            # Test data
            if 'split' in tiles_df.columns:
                test_tiles = tiles_df[tiles_df['split'] == 'test']
            else:
                test_tiles = tiles_df  # Use all data if no splits
            
            self.test_dataset = HistopathDataset(
                tiles_df=test_tiles,
                transform=self.val_transform,
                root_dir=self.data_dir
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return create_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )


class MILDataModule(pl.LightningDataModule):
    """Data module for MIL classification."""
    
    def __init__(
        self,
        data_dir: Path,
        bags_index: str = "bags_index.parquet",
        tiles_index: Optional[str] = "tiles_index.parquet",
        image_size: int = 256,
        batch_size: int = 4,  # Smaller batch size for MIL
        num_workers: int = 4,
        pin_memory: bool = True,
        max_tiles_per_bag: Optional[int] = 1000,
        augment_prob: float = 0.3,  # Lower augmentation for MIL
        normalize: bool = True,
        preload_images: bool = False,
        **kwargs
    ):
        """
        Initialize MIL data module.
        
        Args:
            data_dir: Directory containing processed data
            bags_index: Name of bags index file
            tiles_index: Name of tiles index file (optional)
            image_size: Input image size
            batch_size: Batch size for training (smaller for MIL)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            max_tiles_per_bag: Maximum tiles per bag
            augment_prob: Probability for augmentations
            normalize: Whether to normalize images
            preload_images: Whether to preload all images
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.bags_index = bags_index
        self.tiles_index = tiles_index
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_tiles_per_bag = max_tiles_per_bag
        self.augment_prob = augment_prob
        self.normalize = normalize
        self.preload_images = preload_images
        
        # Will be set in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Transforms
        self.train_transform = HistopathTransforms.get_train_transforms(
            size=image_size,
            normalize=normalize,
            augment_prob=augment_prob
        )
        self.val_transform = HistopathTransforms.get_val_transforms(
            size=image_size,
            normalize=normalize
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        
        # Load bags index
        bags_path = self.data_dir / self.bags_index
        if bags_path.suffix == '.parquet':
            bags_df = pd.read_parquet(bags_path)
        elif bags_path.suffix == '.csv':
            bags_df = pd.read_csv(bags_path)
        else:
            raise ValueError(f"Unsupported bags index format: {bags_path.suffix}")
        
        # Load tiles index if provided
        tiles_df = None
        if self.tiles_index:
            tiles_path = self.data_dir / self.tiles_index
            if tiles_path.exists():
                if tiles_path.suffix == '.parquet':
                    tiles_df = pd.read_parquet(tiles_path)
                elif tiles_path.suffix == '.csv':
                    tiles_df = pd.read_csv(tiles_path)
        
        if stage == "fit" or stage is None:
            # Training data
            if 'split' in bags_df.columns:
                train_bags = bags_df[bags_df['split'] == 'train']
                val_bags = bags_df[bags_df['split'] == 'val']
            else:
                # Fallback: use all data for training, empty validation
                train_bags = bags_df
                val_bags = bags_df.iloc[:0]  # Empty DataFrame
            
            self.train_dataset = MILDataset(
                bags_df=train_bags,
                tiles_df=tiles_df,
                transform=self.train_transform,
                max_tiles_per_bag=self.max_tiles_per_bag,
                root_dir=self.data_dir,
                preload_images=self.preload_images
            )
            
            self.val_dataset = MILDataset(
                bags_df=val_bags,
                tiles_df=tiles_df,
                transform=self.val_transform,
                max_tiles_per_bag=self.max_tiles_per_bag,
                root_dir=self.data_dir,
                preload_images=self.preload_images
            )
        
        if stage == "test" or stage is None:
            # Test data
            if 'split' in bags_df.columns:
                test_bags = bags_df[bags_df['split'] == 'test']
            else:
                test_bags = bags_df  # Use all data if no splits
            
            self.test_dataset = MILDataset(
                bags_df=test_bags,
                tiles_df=tiles_df,
                transform=self.val_transform,
                max_tiles_per_bag=self.max_tiles_per_bag,
                root_dir=self.data_dir,
                preload_images=self.preload_images
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return create_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Get class distribution for each split."""
        distribution = {}
        
        for split in ['train', 'val', 'test']:
            dataset = getattr(self, f"{split}_dataset", None)
            if dataset is not None:
                labels = [dataset.bags_df.iloc[i]['label'] for i in range(len(dataset))]
                distribution[split] = pd.Series(labels).value_counts().to_dict()
        
        return distribution