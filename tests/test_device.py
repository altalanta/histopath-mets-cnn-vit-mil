from __future__ import annotations

import pytest

from histo_omics_lite.utils.device import resolve_device


def test_resolve_device_auto() -> None:
    cfg = resolve_device("auto")
    assert cfg.device.type in {"cpu", "cuda"}


def test_resolve_device_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
