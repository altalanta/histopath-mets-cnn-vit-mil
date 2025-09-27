from __future__ import annotations

from pathlib import Path

from histo_omics_lite.utils.config import load_config


def test_load_config_default(project_root: Path) -> None:
    cfg = load_config(config_name="core", config_path=project_root / "configs")
    assert int(cfg.data.synthetic.splits.train) == 256


def test_load_config_override(project_root: Path) -> None:
    cfg = load_config(config_name="core", config_path=project_root / "configs", overrides=["mode=full"])
    assert int(cfg.data.synthetic.splits.train) == 8192
