"""Inference CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Inference utilities")


@app.command("embed")
def embed(
    data: Path = typer.Argument(..., help="WebDataset shard directory or tar file"),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Trained checkpoint path"),
    out: Path = typer.Option(Path("embeddings.parquet"), "--out", help="Output parquet path"),
    batch_size: int = typer.Option(64, help="Inference batch size"),
    seed: int = typer.Option(0, help="Random seed"),
    device: str = typer.Option("auto", help="Device override (auto|cpu|cuda)"),
    config: Optional[Path] = typer.Option(None, help="Hydra config override"),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override in key=value format (repeatable)",
    ),
) -> None:
    """Batch export multimodal embeddings."""

    from ..inference.embed import export_embeddings

    export_embeddings(
        dataset_path=data,
        checkpoint_path=checkpoint,
        output_path=out,
        batch_size=batch_size,
        seed=seed,
        device_override=device,
        config_path=config,
        overrides=override,
    )
