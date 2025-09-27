"""Training CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Training routines")


def _common_options(
    config: Optional[Path],
    seed: int,
    out: Path,
    device: str,
) -> dict[str, object]:
    return {
        "config_path": config,
        "seed": seed,
        "output_dir": out,
        "device_override": device,
    }


@app.command("simclr")
def train_simclr(
    config: Optional[Path] = typer.Option(None, "--config", help="Hydra config override"),
    seed: int = typer.Option(0, help="Global random seed"),
    out: Path = typer.Option(Path("runs/simclr"), "--out", help="Output directory"),
    device: str = typer.Option("auto", help="Device override (auto|cpu|cuda)"),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override in key=value format (repeatable)",
    ),
) -> None:
    """Run SimCLR pretraining."""

    from ..training.simclr import run_simclr

    run_simclr(**_common_options(config, seed, out, device), overrides=override)


@app.command("clip")
def train_clip(
    config: Optional[Path] = typer.Option(None, "--config", help="Hydra config override"),
    seed: int = typer.Option(0, help="Global random seed"),
    out: Path = typer.Option(Path("runs/clip"), "--out", help="Output directory"),
    device: str = typer.Option("auto", help="Device override (auto|cpu|cuda)"),
    simclr_checkpoint: Optional[Path] = typer.Option(
        None,
        "--simclr-checkpoint",
        help="Optional SimCLR checkpoint to initialise the image encoder",
    ),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override in key=value format (repeatable)",
    ),
) -> None:
    """Run CLIP-style alignment training."""

    from ..training.clip import run_clip

    run_clip(
        **_common_options(config, seed, out, device),
        simclr_checkpoint=simclr_checkpoint,
        overrides=override,
    )
