from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest
import torch
from typer.testing import CliRunner

from histo_omics_lite.cli import app


@pytest.fixture(scope="session")
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _limit_threads() -> Iterator[None]:
    prev_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    prev_env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        torch.set_num_threads(prev_torch_threads)
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
