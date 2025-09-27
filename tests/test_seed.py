from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, strategies as st

from histo_omics_lite.utils.seed import make_worker_init_fn, seed_everything


def test_seed_everything_reproducible() -> None:
    state1 = seed_everything(123)
    tensor1 = torch.randn(3)
    array1 = np.random.rand(3)

    seed_everything(123)
    tensor2 = torch.randn(3)
    array2 = np.random.rand(3)

    torch.testing.assert_close(tensor1, tensor2)
    np.testing.assert_allclose(array1, array2)

    seed_everything(456)
    state2 = seed_everything(123)
    assert state1.python_state == state2.python_state


def test_worker_init_fn_sets_seeds() -> None:
    fn = make_worker_init_fn(100)
    fn(2)
    t1 = torch.randint(0, 10, (1,)).item()
    fn(2)
    t2 = torch.randint(0, 10, (1,)).item()
    assert t1 == t2


@given(st.integers(min_value=0, max_value=1000))
def test_worker_init_property(seed: int) -> None:
    fn = make_worker_init_fn(seed)
    fn(0)
    first = torch.randint(0, 1000, (1,)).item()
    fn(0)
    second = torch.randint(0, 1000, (1,)).item()
    assert first == second
