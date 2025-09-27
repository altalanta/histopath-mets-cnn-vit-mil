from __future__ import annotations

import torch

from histo_omics_lite.evaluation import retrieval


def test_topk_monotonicity() -> None:
    base = torch.eye(4)
    low = retrieval._topk(base * 0.5, k=1)
    high = retrieval._topk(base * 1.5, k=1)
    assert high >= low


def test_bootstrap_ci_bounds() -> None:
    base = torch.eye(4)
    lower, upper = retrieval._bootstrap_ci(
        similarity=base,
        k=1,
        samples=10,
        confidence_level=0.9,
        seed=1,
    )
    assert 0.0 <= lower <= upper <= 1.0
