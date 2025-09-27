"""Evaluation CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Evaluation routines")


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


@app.command("retrieval")
def retrieval(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to CLIP checkpoint"),
    data: Optional[Path] = typer.Option(None, "--data", help="Optional dataset override"),
    config: Optional[Path] = typer.Option(None, "--config", help="Hydra config override"),
    seed: int = typer.Option(0, help="Global random seed"),
    out: Path = typer.Option(Path("reports/retrieval"), "--out", help="Report directory"),
    device: str = typer.Option("auto", help="Device override (auto|cpu|cuda)"),
    override: list[str] = typer.Option(
        [],
        "--override",
        "-o",
        help="Hydra override in key=value format (repeatable)",
    ),
) -> None:
    """Evaluate retrieval performance with bootstrapped CIs and figures."""

    from ..evaluation.retrieval import run_retrieval_eval

    run_retrieval_eval(
        checkpoint_path=checkpoint,
        data_root=data,
        **_common_options(config, seed, out, device),
        overrides=override,
    )
