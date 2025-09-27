from __future__ import annotations

from pathlib import Path

import torch

from histo_omics_lite.utils.profiler import maybe_profile


def test_maybe_profile_disabled(tmp_path: Path) -> None:
    output = tmp_path / "trace.json"
    with maybe_profile(enabled=False, activities=["cpu"], output=output) as prof:
        assert prof is None
    assert not output.exists()


def test_maybe_profile_enabled(tmp_path: Path) -> None:
    output = tmp_path / "trace.json"
    with maybe_profile(enabled=True, activities=["cpu"], output=output) as prof:
        x = torch.randn(2, requires_grad=True)
        y = (x * 2).sum()
        y.backward()
        prof.step()
    assert output.exists()
