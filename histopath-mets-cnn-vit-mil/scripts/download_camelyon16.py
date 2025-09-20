#!/usr/bin/env python3
"""
Download script for CAMELYON16 dataset.

This script provides instructions and validation for downloading the CAMELYON16 dataset.
Due to licensing requirements, automated download is not provided.
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional


# Expected file checksums (MD5) for validation
CAMELYON16_CHECKSUMS = {
    "training": {
        "normal": {
            # These would be actual checksums - using placeholders
            "normal_001.tif": "placeholder_checksum_1",
            "normal_002.tif": "placeholder_checksum_2",
            # ... more files
        },
        "tumor": {
            "tumor_001.tif": "placeholder_checksum_3",
            "tumor_002.tif": "placeholder_checksum_4",
            # ... more files
        },
        "masks": {
            "tumor_001_mask.tif": "placeholder_mask_checksum_1",
            "tumor_002_mask.tif": "placeholder_mask_checksum_2",
            # ... more files
        }
    },
    "testing": {
        "test_001.tif": "placeholder_test_checksum_1",
        "test_002.tif": "placeholder_test_checksum_2",
        # ... more files
    },
    "metadata": {
        "reference.csv": "placeholder_ref_checksum",
        "stage_labels.csv": "placeholder_stage_checksum",
    }
}


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify file checksum matches expected value."""
    if not file_path.exists():
        return False
    
    actual_checksum = calculate_md5(file_path)
    return actual_checksum == expected_checksum


def print_download_instructions() -> None:
    """Print detailed download instructions."""
    print("=" * 80)
    print("CAMELYON16 Dataset Download Instructions")
    print("=" * 80)
    print()
    
    print("⚠️  IMPORTANT: This script does not automatically download the dataset.")
    print("   You must manually download it after agreeing to the terms.")
    print()
    
    print("📋 STEP 1: Review License and Terms")
    print("   - Visit: https://camelyon16.grand-challenge.org/")
    print("   - Read and accept the challenge rules")
    print("   - Register for the challenge if required")
    print()
    
    print("📥 STEP 2: Download Dataset Files")
    print("   Download the following files to your specified output directory:")
    print()
    
    print("   🗂️  Training Data:")
    print("      - normal_XXX.tif (Normal slides)")
    print("      - tumor_XXX.tif (Tumor slides)")
    print("      - tumor_XXX_mask.tif (Annotation masks)")
    print()
    
    print("   🗂️  Test Data:")
    print("      - test_XXX.tif (Test slides)")
    print()
    
    print("   📊 Metadata:")
    print("      - reference.csv (Ground truth labels)")
    print("      - stage_labels.csv (Stage information)")
    print()
    
    print("📁 STEP 3: Organize Files")
    print("   Organize your downloaded files in this structure:")
    print("   data/raw/")
    print("   ├── training/")
    print("   │   ├── normal/")
    print("   │   │   ├── normal_001.tif")
    print("   │   │   └── ...")
    print("   │   ├── tumor/")
    print("   │   │   ├── tumor_001.tif")
    print("   │   │   └── ...")
    print("   │   └── masks/")
    print("   │       ├── tumor_001_mask.tif")
    print("   │       └── ...")
    print("   ├── testing/")
    print("   │   ├── test_001.tif")
    print("   │   └── ...")
    print("   └── metadata/")
    print("       ├── reference.csv")
    print("       └── stage_labels.csv")
    print()
    
    print("✅ STEP 4: Verify Download")
    print("   Run this script again with --verify to check file integrity")
    print()
    
    print("🔍 Dataset Information:")
    print("   - Training slides: 400 (270 normal + 130 tumor)")
    print("   - Test slides: 130")
    print("   - File format: Multi-resolution TIFF")
    print("   - Total size: ~100+ GB")
    print("   - Magnifications: Up to 40x")
    print()
    
    print("📄 Citation:")
    print('   Bejnordi, B.E., et al. "Diagnostic Assessment of Deep Learning')
    print('   Algorithms for Detection of Lymph Node Metastases in Women With')
    print('   Breast Cancer." JAMA 2017.')
    print()


def verify_dataset(data_dir: Path) -> bool:
    """Verify downloaded dataset structure and checksums."""
    print("🔍 Verifying CAMELYON16 dataset...")
    print(f"📁 Data directory: {data_dir}")
    print()
    
    # Check directory structure
    required_dirs = [
        "training/normal",
        "training/tumor", 
        "training/masks",
        "testing",
        "metadata"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    print("✅ Directory structure looks good")
    
    # Count files in each directory
    stats = {}
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if dir_path == "metadata":
            file_count = len([f for f in full_path.glob("*.csv")])
        else:
            file_count = len([f for f in full_path.glob("*.tif")])
        stats[dir_path] = file_count
    
    print("\n📊 File counts:")
    for dir_path, count in stats.items():
        print(f"   {dir_path}: {count} files")
    
    # Verify some checksums (would need actual checksums in production)
    print("\n🔐 Checksum verification:")
    print("   ⚠️  Checksum validation not implemented (placeholder checksums)")
    print("   In production, this would verify file integrity")
    
    # Check minimum expected files
    expected_minimums = {
        "training/normal": 270,
        "training/tumor": 130,
        "training/masks": 130,
        "testing": 130,
        "metadata": 2
    }
    
    all_good = True
    for dir_path, expected_min in expected_minimums.items():
        actual_count = stats.get(dir_path, 0)
        if actual_count < expected_min:
            print(f"❌ {dir_path}: Expected at least {expected_min}, got {actual_count}")
            all_good = False
    
    if all_good:
        print("\n✅ Dataset verification completed successfully!")
        print("   Ready to proceed with data preparation.")
    else:
        print("\n❌ Dataset verification failed!")
        print("   Please check your download and file organization.")
    
    return all_good


def create_readme(output_dir: Path) -> None:
    """Create README file with dataset information."""
    readme_path = output_dir / "README.md"
    
    readme_content = """# CAMELYON16 Dataset

This directory contains the CAMELYON16 dataset for lymph node metastasis detection.

## Dataset Description

The CAMELYON16 dataset contains whole-slide images of sentinel lymph node sections, 
stained with hematoxylin and eosin (H&E). The dataset includes both normal tissue 
and tissue containing breast cancer metastases.

## File Structure

```
data/raw/
├── training/
│   ├── normal/          # Normal tissue slides (270 slides)
│   ├── tumor/           # Tumor tissue slides (130 slides)  
│   └── masks/           # Pixel-level annotations (130 masks)
├── testing/             # Test slides (130 slides)
└── metadata/            # Ground truth and metadata files
    ├── reference.csv    # Ground truth labels
    └── stage_labels.csv # Staging information
```

## Usage

1. **Data Preparation**: Run `python scripts/prepare_camelyon16.py`
2. **Tile Extraction**: Run `python scripts/tile_wsi.py`
3. **MIL Bag Creation**: Run `python scripts/build_mil_bags.py`

## Citation

```bibtex
@article{bejnordi2017diagnostic,
  title={Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer},
  author={Bejnordi, Babak Ehteshami and others},
  journal={JAMA},
  volume={318},
  number={22},
  pages={2199--2210},
  year={2017},
  publisher={American Medical Association}
}
```

## License

Please refer to the CAMELYON16 challenge terms and conditions for usage rights.
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created README at {readme_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="CAMELYON16 dataset download helper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--out", 
        type=Path,
        default="data/raw",
        help="Output directory for dataset (default: data/raw)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset instead of showing download instructions"
    )
    
    parser.add_argument(
        "--create-structure",
        action="store_true", 
        help="Create directory structure for manual file placement"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    if args.verify:
        # Verify existing dataset
        success = verify_dataset(args.out)
        exit(0 if success else 1)
    
    elif args.create_structure:
        # Create directory structure
        dirs_to_create = [
            "training/normal",
            "training/tumor",
            "training/masks", 
            "testing",
            "metadata"
        ]
        
        for dir_path in dirs_to_create:
            (args.out / dir_path).mkdir(parents=True, exist_ok=True)
        
        create_readme(args.out)
        print(f"✅ Created directory structure in {args.out}")
        print("   You can now manually place downloaded files in the appropriate folders")
    
    else:
        # Show download instructions
        print_download_instructions()
        
        print(f"📁 Target directory: {args.out.absolute()}")
        print()
        print("💡 Helpful commands:")
        print(f"   Create structure: python {__file__} --create-structure --out {args.out}")
        print(f"   Verify download:  python {__file__} --verify --out {args.out}")


if __name__ == "__main__":
    main()