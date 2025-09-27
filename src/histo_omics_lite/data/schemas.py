"""Pydantic models and polars schemas for dataset validation."""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, Field, PositiveInt, field_validator

Split = Literal["train", "val"]


class SampleRecord(BaseModel):
    """Schema for individual synthetic samples."""

    sample_id: str = Field(pattern=r"^[a-z]+_[0-9]{6}$")
    split: Split
    label: int
    label_name: str
    shard: str
    key: str
    image_mean: float
    omics_norm: float

    @field_validator("label")
    @classmethod
    def zero_based_label(cls, value: int) -> int:
        if value < 0:
            msg = "Label index must be non-negative"
            raise ValueError(msg)
        return value


class DatasetSummary(BaseModel):
    """Manifest-level metadata summarising the dataset."""

    total_samples: PositiveInt
    train_samples: PositiveInt
    val_samples: PositiveInt
    num_classes: PositiveInt
    image_size: PositiveInt
    transcript_dim: PositiveInt


POLARS_SCHEMA = {
    "sample_id": pl.String,
    "split": pl.Categorical,
    "label": pl.Int32,
    "label_name": pl.Categorical,
    "shard": pl.String,
    "key": pl.String,
    "image_mean": pl.Float32,
    "omics_norm": pl.Float32,
}
