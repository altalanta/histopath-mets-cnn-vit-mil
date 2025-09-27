from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from histo_omics_lite.cli import app


def _override_args(overrides: list[str]) -> list[str]:
    return [arg for override in overrides for arg in ("--override", override)]


def test_cli_help_snapshot(cli_runner: CliRunner, snapshot) -> None:  # type: ignore[type-arg]
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    snapshot.assert_match(result.stdout, "cli_help.txt")


def test_end_to_end_pipeline(tmp_path: Path, cli_runner: CliRunner) -> None:
    dataset_dir = tmp_path / "synthetic"
    simclr_dir = tmp_path / "simclr"
    clip_dir = tmp_path / "clip"
    reports_dir = tmp_path / "reports"

    overrides = [
        f"paths.synthetic_root={dataset_dir}",
        "mode=fast_debug",
        "data.synthetic.image.size=64",
        "data.synthetic.omics.dim=16",
        "data.synthetic.shards.samples_per_shard=16",
    ]

    result = cli_runner.invoke(
        app,
        [
            "data",
            "make-synthetic",
            "--train",
            "24",
            "--val",
            "8",
            "--out",
            str(dataset_dir),
            "--seed",
            "5",
        ]
        + _override_args(overrides),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout

    simclr_overrides = overrides + ["training.simclr.epochs=1", "training.simclr.batch_size=16", "training.simclr.num_workers=0"]
    result = cli_runner.invoke(
        app,
        [
            "train",
            "simclr",
            "--out",
            str(simclr_dir),
            "--seed",
            "5",
        ]
        + _override_args(simclr_overrides),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout

    simclr_ckpt = simclr_dir / "simclr_best.pt"
    assert simclr_ckpt.exists()

    clip_overrides = overrides + [
        "training.clip.epochs=1",
        "training.clip.batch_size=16",
        "training.clip.num_workers=0",
    ]

    result = cli_runner.invoke(
        app,
        [
            "train",
            "clip",
            "--out",
            str(clip_dir),
            "--seed",
            "5",
            "--simclr-checkpoint",
            str(simclr_ckpt),
        ]
        + _override_args(clip_overrides),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout

    clip_ckpt = clip_dir / "clip_best.pt"
    assert clip_ckpt.exists()

    eval_overrides = overrides + ["evaluation.retrieval.bootstrap_samples=8"]
    result = cli_runner.invoke(
        app,
        [
            "eval",
            "retrieval",
            "--checkpoint",
            str(clip_ckpt),
            "--data",
            str(dataset_dir),
            "--out",
            str(reports_dir),
            "--seed",
            "5",
        ]
        + _override_args(eval_overrides),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout

    metrics_path = reports_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "metrics" in metrics and "top1" in metrics["metrics"]

    embedding_path = tmp_path / "embeddings.parquet"
    result = cli_runner.invoke(
        app,
        [
            "infer",
            "embed",
            str(dataset_dir),
            "--checkpoint",
            str(clip_ckpt),
            "--out",
            str(embedding_path),
            "--batch-size",
            "8",
            "--seed",
            "5",
        ]
        + _override_args(overrides),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert embedding_path.exists()
