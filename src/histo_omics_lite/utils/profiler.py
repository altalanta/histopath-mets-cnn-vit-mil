"""Optional torch profiler helper."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Iterable

from torch.profiler import ProfilerActivity, profile, schedule


def _map_activities(names: Iterable[str]) -> list[ProfilerActivity]:
    mapping = {
        "cpu": ProfilerActivity.CPU,
        "cuda": ProfilerActivity.CUDA,
    }
    activities: list[ProfilerActivity] = []
    for name in names:
        lower = name.lower()
        if lower in mapping:
            activities.append(mapping[lower])
    return activities or [ProfilerActivity.CPU]


def maybe_profile(*, enabled: bool, activities: Iterable[str], output: Path) -> AbstractContextManager:
    if not enabled:
        return nullcontext()

    output.parent.mkdir(parents=True, exist_ok=True)

    def _on_trace_ready(profiler) -> None:  # type: ignore[override]
        profiler.export_chrome_trace(str(output))

    return profile(
        activities=_map_activities(activities),
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=_on_trace_ready,
        record_shapes=True,
        with_stack=True,
    )
