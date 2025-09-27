"""Data management CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Data utilities")


@app.command("make-synthetic")
def make_synthetic(
    train: int = typer.Option(256, help="Number of synthetic training samples"),
    val: int = typer.Option(64, help="Number of synthetic validation samples"),
    out: Path = typer.Option(Path("data/synthetic"), help="Output directory"),
    seed: int = typer.Option(0, help="Random seed"),
    device: str = typer.Option("auto", help="Device override (auto|cpu|cuda)"),
    config: Optional[Path] = typer.Option(None, help="Optional Hydra config override"),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override in key=value format (repeatable)",
    ),
) -> None:
    """Generate turnkey synthetic histology/omics WebDataset shards."""

    from ..data.synthetic import generate_synthetic_corpus

    generate_synthetic_corpus(
        num_train=train,
        num_val=val,
        output_dir=out,
        seed=seed,
        device=device,
        config_path=config,
        overrides=override,
    )
