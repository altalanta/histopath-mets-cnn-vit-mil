#!/usr/bin/env python3
"""
Prepare CAMELYON16 dataset by creating metadata and patient splits.

This script processes the raw CAMELYON16 data to create:
- Slide metadata CSV with labels and paths
- Patient-level train/val/test splits
- Quality control reports
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_slide_info(slide_path: Path) -> Dict[str, str]:
    """Extract slide information from filename."""
    stem = slide_path.stem
    
    # Parse different filename patterns
    if stem.startswith("normal_"):
        slide_id = stem.replace("normal_", "")
        slide_type = "normal"
        patient_id = f"patient_{slide_id}"
    elif stem.startswith("tumor_"):
        slide_id = stem.replace("tumor_", "")
        slide_type = "tumor"
        patient_id = f"patient_{slide_id}"
    elif stem.startswith("test_"):
        slide_id = stem.replace("test_", "")
        slide_type = "test"
        patient_id = f"patient_test_{slide_id}"
    else:
        # Try to extract ID from generic pattern
        slide_id = stem
        slide_type = "unknown"
        patient_id = f"patient_{slide_id}"
    
    return {
        "slide_id": slide_id,
        "slide_type": slide_type,
        "patient_id": patient_id,
        "filename": slide_path.name
    }


def create_slide_metadata(raw_data_dir: Path) -> pd.DataFrame:
    """Create slide metadata DataFrame."""
    logging.info("Creating slide metadata...")
    
    slides_data = []
    
    # Process training slides
    for slide_type in ["normal", "tumor"]:
        slide_dir = raw_data_dir / "training" / slide_type
        if not slide_dir.exists():
            logging.warning(f"Directory not found: {slide_dir}")
            continue
            
        for slide_path in slide_dir.glob("*.tif"):
            slide_info = extract_slide_info(slide_path)
            slide_info.update({
                "split": "train",
                "label": 0 if slide_type == "normal" else 1,
                "path": str(slide_path.relative_to(raw_data_dir)),
                "has_mask": slide_type == "tumor"
            })
            
            # Add mask path if available
            if slide_type == "tumor":
                mask_path = raw_data_dir / "training" / "masks" / f"{slide_path.stem}_mask.tif"
                slide_info["mask_path"] = str(mask_path.relative_to(raw_data_dir)) if mask_path.exists() else None
            else:
                slide_info["mask_path"] = None
            
            slides_data.append(slide_info)
    
    # Process test slides
    test_dir = raw_data_dir / "testing"
    if test_dir.exists():
        for slide_path in test_dir.glob("*.tif"):
            slide_info = extract_slide_info(slide_path)
            slide_info.update({
                "split": "test",
                "label": -1,  # Unknown label for test set
                "path": str(slide_path.relative_to(raw_data_dir)),
                "has_mask": False,
                "mask_path": None
            })
            slides_data.append(slide_info)
    
    metadata_df = pd.DataFrame(slides_data)
    
    logging.info(f"Created metadata for {len(metadata_df)} slides")
    logging.info(f"Split distribution: {metadata_df['split'].value_counts().to_dict()}")
    logging.info(f"Label distribution: {metadata_df['label'].value_counts().to_dict()}")
    
    return metadata_df


def load_reference_labels(metadata_dir: Path) -> Dict[str, int]:
    """Load reference labels from CSV if available."""
    reference_file = metadata_dir / "reference.csv"
    
    if not reference_file.exists():
        logging.warning(f"Reference file not found: {reference_file}")
        return {}
    
    try:
        ref_df = pd.read_csv(reference_file)
        # Assuming columns are 'slide_id' and 'label' or similar
        if 'slide_id' in ref_df.columns and 'label' in ref_df.columns:
            return dict(zip(ref_df['slide_id'], ref_df['label']))
        else:
            logging.warning("Reference file does not have expected columns")
            return {}
    except Exception as e:
        logging.error(f"Error loading reference file: {e}")
        return {}


def create_patient_splits(
    metadata_df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """Create patient-level train/val/test splits."""
    logging.info("Creating patient-level splits...")
    
    # Only split training data (test data is already separated)
    train_data = metadata_df[metadata_df['split'] == 'train'].copy()
    
    if len(train_data) == 0:
        logging.warning("No training data found for splitting")
        return metadata_df
    
    # Get unique patients
    patients = train_data[['patient_id', 'label']].drop_duplicates()
    logging.info(f"Found {len(patients)} unique patients in training set")
    
    # First split: train+val vs final_test (but we already have test set)
    # So we split train into train+val
    train_patients, val_patients = train_test_split(
        patients,
        test_size=val_size / (1 - test_size),  # Adjust for remaining data
        stratify=patients['label'],
        random_state=random_state
    )
    
    # Update splits in metadata
    metadata_df = metadata_df.copy()
    
    # Mark validation patients
    val_patient_ids = set(val_patients['patient_id'])
    val_mask = (metadata_df['split'] == 'train') & (metadata_df['patient_id'].isin(val_patient_ids))
    metadata_df.loc[val_mask, 'split'] = 'val'
    
    # Log split statistics
    split_stats = metadata_df.groupby(['split', 'label']).size().unstack(fill_value=0)
    logging.info(f"Final split distribution:\n{split_stats}")
    
    return metadata_df


def validate_data_integrity(metadata_df: pd.DataFrame, raw_data_dir: Path) -> bool:
    """Validate data integrity and file existence."""
    logging.info("Validating data integrity...")
    
    issues = []
    
    # Check file existence
    for idx, row in metadata_df.iterrows():
        file_path = raw_data_dir / row['path']
        if not file_path.exists():
            issues.append(f"Missing file: {file_path}")
        
        # Check mask files
        if row['mask_path'] and pd.notna(row['mask_path']):
            mask_path = raw_data_dir / row['mask_path']
            if not mask_path.exists():
                issues.append(f"Missing mask: {mask_path}")
    
    # Check for duplicate slides
    duplicates = metadata_df[metadata_df.duplicated(['slide_id'], keep=False)]
    if len(duplicates) > 0:
        issues.append(f"Found {len(duplicates)} duplicate slide IDs")
    
    # Check split balance
    train_labels = metadata_df[metadata_df['split'] == 'train']['label']
    if len(train_labels) > 0:
        label_counts = train_labels.value_counts()
        imbalance_ratio = label_counts.max() / label_counts.min() if label_counts.min() > 0 else float('inf')
        if imbalance_ratio > 5:
            issues.append(f"High class imbalance in training set: ratio = {imbalance_ratio:.2f}")
    
    if issues:
        logging.error("Data validation issues found:")
        for issue in issues:
            logging.error(f"  - {issue}")
        return False
    else:
        logging.info("✅ Data validation passed")
        return True


def generate_quality_report(metadata_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate quality control report."""
    logging.info("Generating quality control report...")
    
    report_lines = [
        "# CAMELYON16 Data Quality Report",
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Overview",
        "",
        f"- Total slides: {len(metadata_df)}",
        f"- Training slides: {len(metadata_df[metadata_df['split'] == 'train'])}",
        f"- Validation slides: {len(metadata_df[metadata_df['split'] == 'val'])}",
        f"- Test slides: {len(metadata_df[metadata_df['split'] == 'test'])}",
        "",
        "## Label Distribution",
        "",
    ]
    
    # Add label distribution tables
    for split in ['train', 'val', 'test']:
        split_data = metadata_df[metadata_df['split'] == split]
        if len(split_data) > 0:
            label_dist = split_data['label'].value_counts().sort_index()
            report_lines.extend([
                f"### {split.capitalize()} Set",
                "",
                "| Label | Count | Percentage |",
                "|-------|-------|------------|",
            ])
            
            for label, count in label_dist.items():
                if label >= 0:  # Skip unknown labels (-1)
                    pct = count / len(split_data) * 100
                    label_name = "Normal" if label == 0 else "Tumor"
                    report_lines.append(f"| {label_name} ({label}) | {count} | {pct:.1f}% |")
            
            report_lines.append("")
    
    # Add patient-level statistics
    if 'patient_id' in metadata_df.columns:
        unique_patients = metadata_df['patient_id'].nunique()
        slides_per_patient = metadata_df.groupby('patient_id').size()
        
        report_lines.extend([
            "## Patient-Level Statistics",
            "",
            f"- Unique patients: {unique_patients}",
            f"- Slides per patient (avg): {slides_per_patient.mean():.2f}",
            f"- Slides per patient (range): {slides_per_patient.min()}-{slides_per_patient.max()}",
            "",
        ])
    
    # Write report
    report_path = output_dir / "quality_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Quality report saved to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare CAMELYON16 dataset metadata and splits"
    )
    
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        required=True,
        help="Input directory with raw CAMELYON16 data"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set size (default: 0.15)"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float,
        default=0.15,
        help="Test set size (default: 0.15)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)"
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
    
    logging.info(f"Processing CAMELYON16 data from {args.input_dir}")
    logging.info(f"Output directory: {args.out}")
    
    # Create slide metadata
    metadata_df = create_slide_metadata(args.input_dir)
    
    # Load reference labels if available
    ref_labels = load_reference_labels(args.input_dir / "metadata")
    if ref_labels:
        logging.info(f"Loaded {len(ref_labels)} reference labels")
        # Update test labels if available
        for idx, row in metadata_df.iterrows():
            if row['split'] == 'test' and row['slide_id'] in ref_labels:
                metadata_df.loc[idx, 'label'] = ref_labels[row['slide_id']]
    
    # Create patient splits
    metadata_df = create_patient_splits(
        metadata_df,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Validate data
    is_valid = validate_data_integrity(metadata_df, args.input_dir)
    if not is_valid:
        logging.error("Data validation failed. Please check the issues above.")
        exit(1)
    
    # Save metadata
    metadata_path = args.out / "slide_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logging.info(f"Saved slide metadata to: {metadata_path}")
    
    # Generate quality report
    generate_quality_report(metadata_df, args.out)
    
    # Save split-specific files
    for split in ['train', 'val', 'test']:
        split_data = metadata_df[metadata_df['split'] == split]
        if len(split_data) > 0:
            split_path = args.out / f"{split}_slides.csv"
            split_data.to_csv(split_path, index=False)
            logging.info(f"Saved {split} slides to: {split_path}")
    
    logging.info("✅ Data preparation completed successfully!")


if __name__ == "__main__":
    main()