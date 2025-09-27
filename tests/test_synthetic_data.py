from __future__ import annotations

import json
import tarfile
from pathlib import Path

import polars as pl
import pytest

from histo_omics_lite.data.synthetic import generate_synthetic_corpus


@pytest.mark.parametrize("train,val", [(32, 8)])
def test_generate_synthetic_dataset(tmp_path: Path, train: int, val: int) -> None:
    out_dir = tmp_path / "synthetic"
    generate_synthetic_corpus(
        num_train=train,
        num_val=val,
        output_dir=out_dir,
        seed=7,
        device="cpu",
        config_path=None,
        overrides=[
            "mode=fast_debug",
            "data.synthetic.image.size=64",
            "data.synthetic.omics.dim=16",
            "data.synthetic.shards.samples_per_shard=16",
        ],
    )

    train_shards = sorted((out_dir / "shards" / "train").glob("*.tar*"))
    val_shards = sorted((out_dir / "shards" / "val").glob("*.tar*"))
    assert train_shards and val_shards

    with tarfile.open(train_shards[0], "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        assert any(member.name.endswith(".png") for member in members)
        assert any(member.name.endswith(".npy") for member in members)
        assert any(member.name.endswith(".json") for member in members)

    samples = pl.read_parquet(out_dir / "tables" / "samples.parquet")
    assert samples.height == train + val
    assert set(samples["split"].to_list()) == {"train", "val"}

    manifest = json.loads((out_dir / "manifests" / "dataset.json").read_text())
    assert manifest["summary"]["train_samples"] == train
    assert manifest["summary"]["val_samples"] == val
