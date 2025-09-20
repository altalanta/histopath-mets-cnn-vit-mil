#!/usr/bin/env python3
"""
Build Multiple Instance Learning (MIL) bags from tile data.

This script groups tiles by slide to create bags for MIL training.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_tiles_index(tiles_dir: Path, format: str = "parquet") -> pd.DataFrame:
    """Load tiles index from file."""
    if format == "parquet":
        index_path = tiles_dir / "tiles_index.parquet"
        return pd.read_parquet(index_path)
    elif format == "csv":
        index_path = tiles_dir / "tiles_index.csv"
        return pd.read_csv(index_path)
    elif format == "hdf5":
        index_path = tiles_dir / "tiles_index.h5"
        with h5py.File(index_path, 'r') as f:
            # Load structured array
            tiles_data = f['tiles'][:]
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'tile_id': [t[0].decode() for t in tiles_data],
                'slide_id': [t[1].decode() for t in tiles_data],
                'x': tiles_data['x'],
                'y': tiles_data['y'],
                'level': tiles_data['level'],
                'label': tiles_data['label'],
            })
            
            # Add paths if available
            if 'tile_paths' in f:
                paths = [p.decode() for p in f['tile_paths'][:]]
                df['tile_path'] = paths
            
            return df
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_mil_bags(
    tiles_df: pd.DataFrame,
    min_tiles_per_bag: int = 50,
    max_tiles_per_bag: int = 1000,
    sampling_strategy: str = "random"
) -> pd.DataFrame:
    """
    Create MIL bags from tiles grouped by slide.
    
    Args:
        tiles_df: DataFrame with tile information
        min_tiles_per_bag: Minimum number of tiles per bag
        max_tiles_per_bag: Maximum number of tiles per bag  
        sampling_strategy: How to sample tiles if exceeding max
        
    Returns:
        DataFrame with bag information
    """
    logging.info("Creating MIL bags...")
    
    bags_data = []
    slides_processed = 0
    slides_filtered = 0
    
    # Group tiles by slide
    for slide_id, slide_tiles in tqdm(tiles_df.groupby('slide_id'), desc="Creating bags"):
        n_tiles = len(slide_tiles)
        
        # Filter slides with too few tiles
        if n_tiles < min_tiles_per_bag:
            slides_filtered += 1
            logging.debug(f"Skipping slide {slide_id}: only {n_tiles} tiles (min: {min_tiles_per_bag})")
            continue
        
        # Sample tiles if too many
        if n_tiles > max_tiles_per_bag:
            if sampling_strategy == "random":
                slide_tiles = slide_tiles.sample(n=max_tiles_per_bag, random_state=42)
            elif sampling_strategy == "grid":
                # Sample in a grid pattern
                step = n_tiles // max_tiles_per_bag
                indices = list(range(0, n_tiles, step))[:max_tiles_per_bag]
                slide_tiles = slide_tiles.iloc[indices]
            elif sampling_strategy == "attention_guided":
                # TODO: Implement attention-guided sampling
                # For now, fall back to random
                slide_tiles = slide_tiles.sample(n=max_tiles_per_bag, random_state=42)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Get bag label (should be consistent across tiles)
        bag_label = slide_tiles['label'].iloc[0]
        if not (slide_tiles['label'] == bag_label).all():
            logging.warning(f"Inconsistent labels in slide {slide_id}")
        
        # Create bag metadata
        bag_data = {
            "bag_id": slide_id,
            "slide_id": slide_id,
            "label": bag_label,
            "n_tiles": len(slide_tiles),
            "tile_ids": slide_tiles['tile_id'].tolist(),
            "coords": list(zip(slide_tiles['x'], slide_tiles['y'])),
        }
        
        # Add tile paths if available
        if 'tile_path' in slide_tiles.columns:
            bag_data['tile_paths'] = slide_tiles['tile_path'].tolist()
        
        bags_data.append(bag_data)
        slides_processed += 1
    
    logging.info(f"Created {len(bags_data)} bags from {slides_processed} slides")
    logging.info(f"Filtered out {slides_filtered} slides with insufficient tiles")
    
    return pd.DataFrame(bags_data)


def add_metadata_to_bags(bags_df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """Add slide metadata to bags."""
    if not metadata_path.exists():
        logging.warning(f"Metadata file not found: {metadata_path}")
        return bags_df
    
    slide_metadata = pd.read_csv(metadata_path)
    
    # Merge with bag data
    bags_with_meta = bags_df.merge(
        slide_metadata[['slide_id', 'split', 'patient_id']], 
        on='slide_id', 
        how='left'
    )
    
    return bags_with_meta


def save_bags_data(
    bags_df: pd.DataFrame, 
    output_dir: Path, 
    format: str = "parquet"
) -> None:
    """Save bags data to file."""
    
    if format == "parquet":
        # For parquet, we need to handle list columns carefully
        bags_save = bags_df.copy()
        
        # Convert list columns to string representation
        if 'tile_ids' in bags_save.columns:
            bags_save['tile_ids'] = bags_save['tile_ids'].apply(str)
        if 'coords' in bags_save.columns:
            bags_save['coords'] = bags_save['coords'].apply(str)
        if 'tile_paths' in bags_save.columns:
            bags_save['tile_paths'] = bags_save['tile_paths'].apply(str)
        
        bags_path = output_dir / "bags_index.parquet"
        bags_save.to_parquet(bags_path, index=False)
        
    elif format == "hdf5":
        bags_path = output_dir / "bags_index.h5"
        
        with h5py.File(bags_path, 'w') as f:
            # Save basic bag information
            bag_group = f.create_group('bags')
            
            for idx, row in bags_df.iterrows():
                bag_id = str(row['bag_id'])
                bag_data = bag_group.create_group(bag_id)
                
                # Store metadata
                bag_data.attrs['label'] = row['label']
                bag_data.attrs['n_tiles'] = row['n_tiles']
                bag_data.attrs['slide_id'] = str(row['slide_id'])
                
                if 'split' in row:
                    bag_data.attrs['split'] = str(row['split'])
                if 'patient_id' in row:
                    bag_data.attrs['patient_id'] = str(row['patient_id'])
                
                # Store tile information
                tile_ids = [tid.encode() for tid in row['tile_ids']]
                bag_data.create_dataset('tile_ids', data=tile_ids)
                
                coords = np.array(row['coords'])
                bag_data.create_dataset('coords', data=coords)
                
                if 'tile_paths' in row:
                    tile_paths = [tp.encode() for tp in row['tile_paths']]
                    bag_data.create_dataset('tile_paths', data=tile_paths)
                    
    else:
        # CSV format
        bags_save = bags_df.copy()
        
        # Convert list columns to string representation  
        for col in ['tile_ids', 'coords', 'tile_paths']:
            if col in bags_save.columns:
                bags_save[col] = bags_save[col].apply(lambda x: ';'.join(map(str, x)) if isinstance(x, list) else str(x))
        
        bags_path = output_dir / "bags_index.csv"
        bags_save.to_csv(bags_path, index=False)
    
    logging.info(f"Saved bags data to: {bags_path}")


def generate_bags_summary(bags_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary report for MIL bags."""
    
    report_lines = [
        "# MIL Bags Summary Report",
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- Total bags: {len(bags_df)}",
        f"- Total tiles across all bags: {bags_df['n_tiles'].sum():,}",
        "",
        "## Bag Size Statistics",
        "",
        f"- Mean tiles per bag: {bags_df['n_tiles'].mean():.1f}",
        f"- Median tiles per bag: {bags_df['n_tiles'].median():.0f}",
        f"- Min tiles per bag: {bags_df['n_tiles'].min()}",
        f"- Max tiles per bag: {bags_df['n_tiles'].max()}",
        f"- Std tiles per bag: {bags_df['n_tiles'].std():.1f}",
        "",
    ]
    
    # Add distribution by split if available
    if 'split' in bags_df.columns:
        split_summary = bags_df.groupby(['split', 'label']).agg({
            'bag_id': 'count',
            'n_tiles': ['sum', 'mean']
        }).round(1)
        
        report_lines.extend([
            "## Distribution by Split",
            "",
            "| Split | Label | N_Bags | Total_Tiles | Avg_Tiles |",
            "|-------|-------|--------|-------------|-----------|",
        ])
        
        for (split, label), row in split_summary.iterrows():
            label_name = "Normal" if label == 0 else "Tumor"
            n_bags = row[('bag_id', 'count')]
            total_tiles = row[('n_tiles', 'sum')]
            avg_tiles = row[('n_tiles', 'mean')]
            report_lines.append(f"| {split} | {label_name} | {n_bags} | {total_tiles:,} | {avg_tiles:.1f} |")
        
        report_lines.append("")
    
    # Add label distribution
    label_dist = bags_df['label'].value_counts().sort_index()
    
    report_lines.extend([
        "## Label Distribution",
        "",
        "| Label | Count | Percentage |",
        "|-------|-------|------------|",
    ])
    
    for label, count in label_dist.items():
        label_name = "Normal" if label == 0 else "Tumor" if label == 1 else f"Label_{label}"
        pct = count / len(bags_df) * 100
        report_lines.append(f"| {label_name} | {count} | {pct:.1f}% |")
    
    # Write report
    report_path = output_dir / "bags_summary.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Bags summary saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build MIL bags from tile data"
    )
    
    parser.add_argument(
        "--tiles",
        type=Path,
        required=True,
        help="Directory containing tiles index"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for bags data"
    )
    
    parser.add_argument(
        "--min-tiles",
        type=int,
        default=50,
        help="Minimum tiles per bag (default: 50)"
    )
    
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=1000,
        help="Maximum tiles per bag (default: 1000)"
    )
    
    parser.add_argument(
        "--sampling",
        choices=["random", "grid", "attention_guided"],
        default="random",
        help="Sampling strategy for large bags (default: random)"
    )
    
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "hdf5"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    
    parser.add_argument(
        "--tiles-format",
        choices=["parquet", "csv", "hdf5"],
        default="parquet",
        help="Input tiles format (default: parquet)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Load tiles index
    logging.info(f"Loading tiles index from: {args.tiles}")
    tiles_df = load_tiles_index(args.tiles, args.tiles_format)
    logging.info(f"Loaded {len(tiles_df)} tiles from {tiles_df['slide_id'].nunique()} slides")
    
    # Create MIL bags
    bags_df = create_mil_bags(
        tiles_df,
        min_tiles_per_bag=args.min_tiles,
        max_tiles_per_bag=args.max_tiles,
        sampling_strategy=args.sampling
    )
    
    # Add metadata if available
    metadata_path = args.tiles.parent / "processed" / "slide_metadata.csv"
    bags_df = add_metadata_to_bags(bags_df, metadata_path)
    
    # Save bags data
    save_bags_data(bags_df, args.out, args.format)
    
    # Generate summary
    generate_bags_summary(bags_df, args.out)
    
    # Save split-specific bag files
    if 'split' in bags_df.columns:
        for split in bags_df['split'].unique():
            if pd.notna(split):
                split_bags = bags_df[bags_df['split'] == split]
                split_path = args.out / f"{split}_bags.csv"
                
                # Save simplified version for quick loading
                split_simple = split_bags[['bag_id', 'slide_id', 'label', 'n_tiles']].copy()
                split_simple.to_csv(split_path, index=False)
                logging.info(f"Saved {split} bags to: {split_path}")
    
    logging.info("✅ MIL bags creation completed successfully!")


if __name__ == "__main__":
    main()