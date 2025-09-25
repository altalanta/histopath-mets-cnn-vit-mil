"""Inference utilities for generating embeddings."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import HistopathDataset
from .data.transforms import HistopathTransforms
from .utils.seed import seed_everything

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for inference."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    map_location: Optional[str] = None,
) -> torch.nn.Module:
    """Load model from checkpoint."""
    if device is None:
        device = get_device()
    
    if map_location is None:
        map_location = str(device)
    
    try:
        # Try to load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            # Assume the checkpoint is the state dict directly
            state_dict = checkpoint
        
        # TODO: This would need to be replaced with actual model creation
        # For now, create a dummy model structure
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This would be replaced with actual model architecture
                self.features = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(1024, 512)  # Dummy embedding size
                )
            
            def forward(self, x):
                return self.features(x)
        
        model = DummyModel()
        
        # Try to load state dict (this might fail with dummy model)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Could not load state dict: {e}")
            logger.warning("Using randomly initialized model for demonstration")
        
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise


def create_inference_dataloader(
    tiles_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> DataLoader:
    """Create dataloader for inference."""
    
    # Create dataset with validation transforms (no augmentation)
    transform = HistopathTransforms.get_val_transforms(
        size=image_size, 
        normalize=True
    )
    
    dataset = HistopathDataset(
        tiles_df=tiles_df,
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for inference
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """Extract embeddings from model."""
    
    model.eval()
    
    all_embeddings = []
    all_sample_ids = []
    all_slide_ids = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move to device
            images = batch["image"].to(device)
            
            # Forward pass
            embeddings = model(images)
            
            # Ensure embeddings are 2D
            if embeddings.dim() > 2:
                embeddings = F.adaptive_avg_pool2d(embeddings, (1, 1)).flatten(1)
            
            # L2 normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Store results
            all_embeddings.append(embeddings.cpu().numpy())
            
            if include_metadata:
                all_sample_ids.extend(batch.get("tile_id", [f"tile_{i}" for i in range(len(images))]))
                all_slide_ids.extend(batch.get("slide_id", ["unknown"] * len(images)))
                all_labels.extend(batch.get("label", [-1] * len(images)))
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    # Create DataFrame
    data = {}
    
    # Add embedding dimensions
    embedding_dim = embeddings_array.shape[1]
    for i in range(embedding_dim):
        data[f"emb_{i:04d}"] = embeddings_array[:, i]
    
    # Add metadata if requested
    if include_metadata:
        data["tile_id"] = all_sample_ids
        data["slide_id"] = all_slide_ids
        data["label"] = all_labels
    
    df = pd.DataFrame(data)
    
    # Set tile_id as index if available
    if "tile_id" in df.columns:
        df.set_index("tile_id", inplace=True)
    
    return df


def discover_tiles(tiles_path: Path) -> pd.DataFrame:
    """Discover tile files and create metadata DataFrame."""
    
    if tiles_path.is_file():
        # Handle tar files or single file
        if tiles_path.suffix.lower() in ['.tar', '.tar.gz']:
            # TODO: Implement tar file handling
            raise NotImplementedError("Tar file handling not yet implemented")
        else:
            raise ValueError(f"Single file input not supported: {tiles_path}")
    
    elif tiles_path.is_dir():
        # Discover image files in directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
        
        tiles_data = []
        
        for img_path in tiles_path.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Extract metadata from path/filename
                tile_id = img_path.stem
                
                # Try to extract slide_id from path structure
                # Assume structure like: slides/slide_001/tile_001.png
                path_parts = img_path.parts
                if len(path_parts) >= 2:
                    slide_id = path_parts[-2]  # Parent directory
                else:
                    slide_id = "unknown"
                
                # Default label (would need to be provided separately in real use)
                label = 0  # Binary classification default
                
                tiles_data.append({
                    "tile_path": str(img_path),
                    "tile_id": tile_id,
                    "slide_id": slide_id,
                    "label": label,
                })
        
        if not tiles_data:
            raise ValueError(f"No image files found in {tiles_path}")
        
        df = pd.DataFrame(tiles_data)
        logger.info(f"Discovered {len(df)} tiles in {len(df['slide_id'].unique())} slides")
        
        return df
    
    else:
        raise ValueError(f"Invalid tiles path: {tiles_path}")


def run_embed_pipeline(
    tiles_dir: Path,
    model_path: Path,
    config_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = "auto",
    include_metadata: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    image_size: int = 224,
) -> pd.DataFrame:
    """
    Run complete embedding generation pipeline.
    
    Args:
        tiles_dir: Directory containing tile images
        model_path: Path to model checkpoint
        config_path: Optional path to configuration file
        batch_size: Batch size for inference
        device: Device to use (auto, cpu, cuda)
        include_metadata: Whether to include metadata in output
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        image_size: Input image size
        
    Returns:
        DataFrame with embeddings and metadata
    """
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Get device
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")
    
    # Discover tiles
    logger.info("Discovering tiles...")
    tiles_df = discover_tiles(tiles_dir)
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = load_model_from_checkpoint(model_path, device=device_obj)
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_inference_dataloader(
        tiles_df=tiles_df,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings_df = extract_embeddings(
        model=model,
        dataloader=dataloader,
        device=device_obj,
        include_metadata=include_metadata,
    )
    
    logger.info(f"Generated embeddings: {embeddings_df.shape}")
    
    return embeddings_df


# Public API for library use
def embed(
    tiles_path: Union[str, Path],
    model_path: Union[str, Path],
    batch_size: int = 32,
    device: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Public API for embedding generation.
    
    Args:
        tiles_path: Path to tiles directory or tar file
        model_path: Path to model checkpoint
        batch_size: Batch size for inference
        device: Device to use
        seed: Random seed
        
    Returns:
        DataFrame with embeddings
    """
    return run_embed_pipeline(
        tiles_dir=Path(tiles_path),
        model_path=Path(model_path),
        batch_size=batch_size,
        device=device,
        seed=seed,
    )