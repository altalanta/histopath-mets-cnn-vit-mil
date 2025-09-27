"""Hydra configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def is_hydra_config(path: Path) -> bool:
    return path.suffix in {".yaml", ".yml"}


def load_config(
    *,
    config_name: str,
    config_path: Optional[Path] = None,
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load a Hydra configuration, allowing per-call overrides."""

    if config_path is None:
        search_dir = CONFIG_DIR
        name = config_name
    elif config_path.is_dir():
        search_dir = config_path
        name = config_name
    elif is_hydra_config(config_path):
        search_dir = config_path.parent
        name = config_path.stem
    else:
        raise FileNotFoundError(f"Unsupported config path: {config_path}")

    if not search_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {search_dir}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(search_dir), job_name="histo-omics-lite"):
        cfg = compose(config_name=name, overrides=list(overrides or []))
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, object]:
    """Convert a Hydra DictConfig to a plain dictionary."""

    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def flatten_overrides(items: Iterable[str]) -> list[str]:
    """Normalize override strings, filtering empties."""

    return [item for item in items if item]
