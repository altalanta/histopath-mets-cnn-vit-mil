#!/usr/bin/env python3
"""
Tile whole slide images into patches for model training.

This script extracts tiles from WSI files and creates an index for efficient loading.
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from histopath.utils.tiling import WSITiler
from histopath.utils.tissue_mask import create_tissue_mask, filter_tile_by_tissue_content


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_tissue_mask_for_slide(slide_path: Path, downsample: int = 32) -> Optional[np.ndarray]:
    """Create tissue mask for a whole slide image."""
    try:
        import openslide
        
        slide = openslide.OpenSlide(str(slide_path))
        
        # Get appropriate level for tissue detection
        level_count = slide.level_count
        level = min(level_count - 1, 2)  # Use level 2 or highest available
        
        # Read thumbnail
        thumbnail = slide.read_region(
            (0, 0), level, slide.level_dimensions[level]
        ).convert('RGB')
        
        slide.close()
        
        # Create tissue mask
        thumbnail_array = np.array(thumbnail)
        tissue_mask = create_tissue_mask(
            thumbnail_array,
            method="otsu",
            min_tissue_ratio=0.1
        )
        
        return tissue_mask
        
    except Exception as e:
        logging.error(f"Error creating tissue mask for {slide_path}: {e}")
        return None


def process_single_slide(
    slide_info: Tuple[str, str, int, Path, Path],
    tiler: WSITiler,
    save_images: bool = True,
    tissue_filter: bool = True
) -> List[Dict]:
    """Process a single slide and extract tiles."""
    slide_id, slide_path, label, output_dir, temp_dir = slide_info
    
    try:
        slide_path = Path(slide_path)
        if not slide_path.exists():
            logging.error(f"Slide not found: {slide_path}")
            return []
        
        logging.info(f"Processing slide: {slide_id}")
        
        # Create tissue mask if filtering enabled
        tissue_mask = None
        if tissue_filter:
            tissue_mask = create_tissue_mask_for_slide(slide_path)
            if tissue_mask is None:
                logging.warning(f"Failed to create tissue mask for {slide_id}, skipping tissue filtering")
        
        # Get tile coordinates
        coordinates = tiler.get_tile_coordinates(str(slide_path), tissue_mask)
        
        if len(coordinates) == 0:
            logging.warning(f"No valid tiles found for slide {slide_id}")
            return []
        
        logging.info(f"Found {len(coordinates)} tile coordinates for {slide_id}")
        
        # Extract tiles
        tiles_data = []
        
        for i, (x, y, level) in enumerate(coordinates):
            try:
                # Extract tile
                tile = tiler.extract_tile(str(slide_path), x, y, level)
                
                # Additional tissue filtering at tile level
                if tissue_filter:
                    tile_array = np.array(tile)
                    if not filter_tile_by_tissue_content(tile_array, tissue_threshold=0.3):
                        continue
                
                # Generate tile ID
                tile_id = f"{slide_id}_x{x}_y{y}_l{level}"
                
                # Save tile image if requested
                tile_path = None
                if save_images:
                    # Create subdirectory for slide
                    slide_output_dir = output_dir / "images" / slide_id
                    slide_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    tile_filename = f"{tile_id}.png"
                    tile_path = slide_output_dir / tile_filename
                    tile.save(tile_path, "PNG")
                
                # Store tile metadata
                tile_data = {
                    "tile_id": tile_id,
                    "slide_id": slide_id,
                    "x": x,
                    "y": y,
                    "level": level,
                    "label": label,
                    "tile_path": str(tile_path.relative_to(output_dir)) if tile_path else None,
                    "slide_path": str(slide_path),
                    "width": tile.width,
                    "height": tile.height,
                }
                
                tiles_data.append(tile_data)
                
            except Exception as e:
                logging.error(f"Error processing tile {i} for slide {slide_id}: {e}")
                continue
        
        logging.info(f"Successfully processed {len(tiles_data)} tiles for slide {slide_id}")
        return tiles_data
        
    except Exception as e:
        logging.error(f"Error processing slide {slide_id}: {e}")
        return []


def process_slides_parallel(
    slides_df: pd.DataFrame,
    tiler: WSITiler,
    output_dir: Path,
    num_workers: int = 4,
    save_images: bool = True,
    tissue_filter: bool = True
) -> pd.DataFrame:
    """Process multiple slides in parallel."""
    
    # Prepare arguments for parallel processing
    slide_args = []
    for _, row in slides_df.iterrows():
        slide_args.append((
            row['slide_id'],
            row['slide_path'],
            row['label'],
            output_dir,
            output_dir / "temp"
        ))
    
    # Create temporary directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    all_tiles_data = []
    
    if num_workers == 1:
        # Sequential processing
        for slide_info in tqdm(slide_args, desc="Processing slides"):
            tiles_data = process_single_slide(
                slide_info, tiler, save_images, tissue_filter
            )
            all_tiles_data.extend(tiles_data)
    else:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            # Create partial function with fixed arguments
            process_func = lambda slide_info: process_single_slide(
                slide_info, tiler, save_images, tissue_filter
            )
            
            # Process slides in parallel
            results = pool.map(process_func, slide_args)
            
            # Flatten results
            for tiles_data in results:
                all_tiles_data.extend(tiles_data)
    
    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass
    
    return pd.DataFrame(all_tiles_data)


def save_tiles_index(tiles_df: pd.DataFrame, output_dir: Path, format: str = "parquet") -> None:
    """Save tiles index to file."""
    if format == "parquet":
        index_path = output_dir / "tiles_index.parquet"
        tiles_df.to_parquet(index_path, index=False)
    elif format == "csv":
        index_path = output_dir / "tiles_index.csv"
        tiles_df.to_csv(index_path, index=False)
    elif format == "hdf5":
        index_path = output_dir / "tiles_index.h5"
        with h5py.File(index_path, 'w') as f:
            # Save as structured array
            dt = np.dtype([
                ('tile_id', 'S50'),
                ('slide_id', 'S20'),
                ('x', np.int32),
                ('y', np.int32),
                ('level', np.int32),
                ('label', np.int32),
            ])
            
            data = np.array([
                (row['tile_id'].encode(), row['slide_id'].encode(), 
                 row['x'], row['y'], row['level'], row['label'])
                for _, row in tiles_df.iterrows()
            ], dtype=dt)
            
            f.create_dataset('tiles', data=data)
            
            # Save string columns separately
            if 'tile_path' in tiles_df.columns:
                paths = [str(p).encode() for p in tiles_df['tile_path']]
                f.create_dataset('tile_paths', data=paths)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Saved tiles index to: {index_path}")


def generate_summary_report(tiles_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary report of tiling process."""
    
    report_lines = [
        "# Tiling Summary Report",
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- Total tiles extracted: {len(tiles_df):,}",
        f"- Unique slides processed: {tiles_df['slide_id'].nunique()}",
        "",
        "## Distribution by Split",
        "",
    ]
    
    # Add distribution tables
    if 'slide_id' in tiles_df.columns:
        # Load slide metadata to get splits
        metadata_path = output_dir.parent / "slide_metadata.csv"
        if metadata_path.exists():
            slide_metadata = pd.read_csv(metadata_path)
            tiles_with_split = tiles_df.merge(
                slide_metadata[['slide_id', 'split']], 
                on='slide_id', 
                how='left'
            )
            
            split_summary = tiles_with_split.groupby(['split', 'label']).size().unstack(fill_value=0)
            
            report_lines.extend([
                "| Split | Normal | Tumor | Total |",
                "|-------|--------|-------|-------|",
            ])
            
            for split in split_summary.index:
                normal = split_summary.loc[split, 0] if 0 in split_summary.columns else 0
                tumor = split_summary.loc[split, 1] if 1 in split_summary.columns else 0
                total = normal + tumor
                report_lines.append(f"| {split} | {normal:,} | {tumor:,} | {total:,} |")
    
    # Add per-slide statistics
    tiles_per_slide = tiles_df.groupby('slide_id').size()
    
    report_lines.extend([
        "",
        "## Tiles per Slide Statistics",
        "",
        f"- Mean: {tiles_per_slide.mean():.1f}",
        f"- Median: {tiles_per_slide.median():.0f}",
        f"- Min: {tiles_per_slide.min()}",
        f"- Max: {tiles_per_slide.max()}",
        f"- Std: {tiles_per_slide.std():.1f}",
        "",
    ])
    
    # Write report
    report_path = output_dir / "tiling_summary.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Summary report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract tiles from whole slide images"
    )
    
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input directory with processed slide metadata"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for tiles"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between tiles in pixels (default: 0)"
    )
    
    parser.add_argument(
        "--magnification",
        type=int,
        default=10,
        choices=[5, 10, 20, 40],
        help="Target magnification (default: 10)"
    )
    
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=10000,
        help="Maximum tiles per slide (default: 10000)"
    )
    
    parser.add_argument(
        "--otsu-mask",
        action="store_true",
        help="Use Otsu thresholding for tissue detection"
    )
    
    parser.add_argument(
        "--no-tissue-filter",
        action="store_true",
        help="Disable tissue content filtering"
    )
    
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save individual tile images (index only)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "hdf5"],
        default="parquet",
        help="Output format for tiles index (default: parquet)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load slide metadata
    metadata_path = args.input_dir / "slide_metadata.csv"
    if not metadata_path.exists():
        logging.error(f"Slide metadata not found: {metadata_path}")
        logging.error("Please run prepare_camelyon16.py first")
        exit(1)
    
    slides_df = pd.read_csv(metadata_path)
    logging.info(f"Loaded metadata for {len(slides_df)} slides")
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Initialize tiler
    tiler = WSITiler(
        tile_size=args.tile_size,
        overlap=args.overlap,
        magnification=args.magnification,
        max_tiles_per_slide=args.max_tiles
    )
    
    # Convert relative paths to absolute
    slides_df['slide_path'] = slides_df['path'].apply(
        lambda p: str(args.input_dir.parent / "raw" / p)
    )
    
    # Filter to only slides that exist
    existing_slides = []
    for _, row in slides_df.iterrows():
        if Path(row['slide_path']).exists():
            existing_slides.append(row)
        else:
            logging.warning(f"Slide file not found: {row['slide_path']}")
    
    if not existing_slides:
        logging.error("No slide files found")
        exit(1)
    
    slides_df = pd.DataFrame(existing_slides)
    logging.info(f"Processing {len(slides_df)} existing slides")
    
    # Process slides
    tiles_df = process_slides_parallel(
        slides_df,
        tiler,
        args.out,
        num_workers=args.workers,
        save_images=not args.no_save_images,
        tissue_filter=not args.no_tissue_filter
    )
    
    if len(tiles_df) == 0:
        logging.error("No tiles were extracted")
        exit(1)
    
    logging.info(f"Extracted {len(tiles_df)} tiles total")
    
    # Save tiles index
    save_tiles_index(tiles_df, args.out, args.format)
    
    # Generate summary report
    generate_summary_report(tiles_df, args.out)
    
    logging.info("✅ Tiling completed successfully!")


if __name__ == "__main__":
    main()