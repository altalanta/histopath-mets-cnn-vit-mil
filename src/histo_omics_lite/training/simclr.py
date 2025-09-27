"""SimCLR training entrypoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console

from ..data.pipeline import build_simclr_dataloader
from ..data.transforms import SimCLRTransformConfig
from ..utils.config import load_config, to_dict
from ..utils.device import resolve_device
from ..utils.profiler import maybe_profile
from ..utils.seed import seed_everything
from .models import ProjectionConfig, SimCLRModel, simclr_nt_xent

console = Console()


def _device_choice(config_device: str, override: str) -> str:
    if override != "auto":
        return override
    return config_device


def run_simclr(
    *,
    config_path: Optional[Path],
    seed: int,
    output_dir: Path,
    device_override: str,
    overrides: Optional[list[str]] = None,
) -> None:
    cfg = load_config(config_name="core", config_path=config_path, overrides=overrides or [])
    seed_everything(seed)

    device_choice = _device_choice(str(cfg.device), device_override)
    device_cfg = resolve_device(device_choice)
    device = device_cfg.device

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    simclr_cfg = cfg.training.simclr
    data_cfg = cfg.data.synthetic

    transform_cfg = SimCLRTransformConfig(
        image_size=int(data_cfg.image.size),
        color_jitter=float(data_cfg.augment.color_jitter),
        blur_prob=float(data_cfg.augment.blur_prob),
        rotation=int(data_cfg.augment.rotation),
    )

    batch_size = int(simclr_cfg.batch_size)

    dataloader = build_simclr_dataloader(
        root=Path(cfg.paths.synthetic_root),
        batch_size=batch_size,
        num_workers=int(simclr_cfg.num_workers),
        seed=seed,
        device_cfg=device_cfg,
        transform_cfg=transform_cfg,
    )

    model = SimCLRModel(
        ProjectionConfig(
            hidden_dim=int(simclr_cfg.hidden_dim),
            projection_dim=int(simclr_cfg.projection_dim),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(simclr_cfg.base_learning_rate), weight_decay=float(simclr_cfg.weight_decay))

    temperature = float(simclr_cfg.temperature)
    log_interval = int(simclr_cfg.log_interval)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = output_dir / "simclr_best.pt"

    profiler_path = output_dir / "profiler_trace.json"

    epochs = int(simclr_cfg.epochs)
    console.log(f"Starting SimCLR for {epochs} epochs", device=str(device))

    with maybe_profile(
        enabled=bool(cfg.profiling.enabled),
        activities=list(cfg.profiling.activities),
        output=profiler_path,
    ) as profiler:
        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_start = time.perf_counter()
            for batch_idx, batch in enumerate(dataloader, start=1):
                view1 = batch["view1"].to(device)
                view2 = batch["view2"].to(device)

                _, proj1 = model(view1)
                _, proj2 = model(view2)
                loss = simclr_nt_xent(proj1, proj2, temperature)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                global_step += 1

                if profiler is not None:
                    profiler.step()

                if batch_idx % log_interval == 0:
                    elapsed = time.perf_counter() - batch_start
                    samples = log_interval * batch_size
                    throughput = samples / elapsed if elapsed > 0 else 0.0
                    console.log(
                        f"epoch={epoch + 1} step={batch_idx}",
                        loss=f"{loss.item():.4f}",
                        throughput=f"{throughput:0.1f} img/s",
                    )
                    batch_start = time.perf_counter()

            avg_loss = epoch_loss / max(1, len(dataloader))
            history.append({"epoch": epoch + 1, "loss": avg_loss})
            console.log(f"epoch {epoch + 1} | loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": to_dict(cfg),
                    },
                    best_path,
                )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    console.log(f"SimCLR complete. Best checkpoint: {best_path}", loss=f"{best_loss:.4f}")
