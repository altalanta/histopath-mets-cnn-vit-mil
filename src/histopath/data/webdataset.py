"""Robust WebDataset implementation with shard validation and atomic writes."""

import io
import os
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset


class RobustWebDataset(IterableDataset):
    """WebDataset with robustness features for histopathology data."""

    def __init__(
        self,
        urls: Union[str, List[str]],
        shuffle_seed: int = 42,
        max_shard_size: int = 1000,
        max_shard_bytes: int = 100 * 1024 * 1024,  # 100MB
        validate_shards: bool = True,
        transform: Optional[Any] = None,
    ):
        """
        Initialize robust WebDataset.

        Args:
            urls: URL(s) or path(s) to webdataset shards
            shuffle_seed: Seed for deterministic shuffling
            max_shard_size: Maximum number of samples per shard
            max_shard_bytes: Maximum bytes per shard
            validate_shards: Whether to validate shard integrity
            transform: Optional transforms to apply
        """
        self.urls = urls if isinstance(urls, list) else [urls]
        self.shuffle_seed = shuffle_seed
        self.max_shard_size = max_shard_size
        self.max_shard_bytes = max_shard_bytes
        self.validate_shards = validate_shards
        self.transform = transform
        self._rng = np.random.default_rng(shuffle_seed)

        # Validate all shards at initialization if requested
        if self.validate_shards:
            self._validate_all_shards()

    def _validate_all_shards(self) -> None:
        """Validate all shards for corruption."""
        for url in self.urls:
            try:
                self.validate_shard(url)
            except RuntimeError as e:
                warnings.warn(f"Shard validation failed: {e}")

    def validate_shard(self, shard_path: str) -> None:
        """
        Validate shard integrity.

        Args:
            shard_path: Path to shard file

        Raises:
            RuntimeError: If shard is corrupted
        """
        try:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if member.isfile():
                        # Try to extract a small portion to verify integrity
                        data = tar.extractfile(member)
                        if data is None:
                            raise RuntimeError(f"Cannot extract member {member.name}")
                        # Read first 1024 bytes to verify
                        try:
                            data.read(1024)
                        except Exception as e:
                            raise RuntimeError(f"Cannot read member {member.name}: {e}")
        except (tarfile.TarError, OSError) as e:
            raise RuntimeError(f"Corrupted shard {shard_path}: {e}")

    def write_shard_atomic(
        self,
        final_path: str,
        members: Iterator[Dict[str, Any]],
        max_records: Optional[int] = None,
    ) -> int:
        """
        Write shard atomically to prevent partial files.

        Args:
            final_path: Final path for the shard
            members: Iterator of member data dictionaries
            max_records: Maximum records to write (uses max_shard_size if None)

        Returns:
            Number of records written
        """
        if max_records is None:
            max_records = self.max_shard_size

        final_path = Path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file in same directory for atomic rename
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_", suffix=".tar", dir=final_path.parent
        )
        os.close(fd)

        try:
            written_count = 0
            current_bytes = 0

            with tarfile.open(tmp_path, "w") as tar:
                for member_data in members:
                    if written_count >= max_records:
                        break

                    # Check size constraints
                    if current_bytes >= self.max_shard_bytes:
                        break

                    # Write each component of the sample
                    for key, data in member_data.items():
                        if isinstance(data, bytes):
                            # Write binary data
                            info = tarfile.TarInfo(
                                name=f"{written_count:06d}.{key}"
                            )
                            info.size = len(data)
                            tar.addfile(info, fileobj=io.BytesIO(data))
                            current_bytes += len(data)
                        elif isinstance(data, str):
                            # Write text data
                            data_bytes = data.encode('utf-8')
                            info = tarfile.TarInfo(
                                name=f"{written_count:06d}.{key}"
                            )
                            info.size = len(data_bytes)
                            tar.addfile(info, fileobj=io.BytesIO(data_bytes))
                            current_bytes += len(data_bytes)

                    written_count += 1

            # Atomic rename
            os.replace(tmp_path, final_path)
            return written_count

        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

    def __iter__(self):
        """Iterate over dataset with deterministic shuffling."""
        # Create shuffled list of URLs
        urls_shuffled = self.urls.copy()
        self._rng.shuffle(urls_shuffled)

        for shard_url in urls_shuffled:
            try:
                # Validate shard before reading if requested
                if self.validate_shards:
                    self.validate_shard(shard_url)

                with tarfile.open(shard_url, "r") as tar:
                    # Group members by sample ID
                    samples = {}
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue

                        # Parse sample ID from filename (assumes format: XXXXXX.ext)
                        name_parts = member.name.split(".")
                        if len(name_parts) >= 2:
                            sample_id = name_parts[0]
                            extension = ".".join(name_parts[1:])

                            if sample_id not in samples:
                                samples[sample_id] = {}

                            # Extract data
                            data_file = tar.extractfile(member)
                            if data_file:
                                samples[sample_id][extension] = data_file.read()

                    # Yield complete samples
                    sample_ids = list(samples.keys())
                    self._rng.shuffle(sample_ids)  # Shuffle samples within shard

                    for sample_id in sample_ids:
                        sample_data = samples[sample_id]

                        # Apply transform if provided
                        if self.transform:
                            sample_data = self.transform(sample_data)

                        yield sample_data

            except Exception as e:
                warnings.warn(f"Error reading shard {shard_url}: {e}")
                continue


def create_validation_script():
    """Create standalone shard validation script."""
    validation_script = '''#!/usr/bin/env python3
"""Standalone shard validation script."""

import sys
import tarfile
from pathlib import Path

def validate_shard(shard_path: str) -> bool:
    """Validate a single shard file."""
    try:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    data = tar.extractfile(member)
                    if data is None:
                        print(f"ERROR: Cannot extract {member.name} from {shard_path}")
                        return False
                    # Try to read the data
                    try:
                        data.read()
                    except Exception as e:
                        print(f"ERROR: Cannot read {member.name}: {e}")
                        return False
        print(f"OK: {shard_path}")
        return True
    except Exception as e:
        print(f"ERROR: {shard_path} - {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_shards.py <shard1> [shard2] ...")
        sys.exit(1)
    
    failed = 0
    for shard_path in sys.argv[1:]:
        if not validate_shard(shard_path):
            failed += 1
    
    if failed > 0:
        print(f"FAILED: {failed} shard(s) corrupted")
        sys.exit(1)
    else:
        print("All shards validated successfully")

if __name__ == "__main__":
    main()
'''
    
    # Write validation script
    script_path = Path(__file__).parent / "validate_shards.py"
    with open(script_path, 'w') as f:
        f.write(validation_script)
    
    # Make executable
    import stat
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    
    return script_path


if __name__ == "__main__":
    # Create validation script when module is run
    script_path = create_validation_script()
    print(f"Created validation script: {script_path}")