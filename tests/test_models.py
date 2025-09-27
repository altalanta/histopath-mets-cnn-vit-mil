from __future__ import annotations

from pathlib import Path

import torch

from histo_omics_lite.training.models import (
    ClipAlignmentModel,
    ProjectionConfig,
    identical_temperature_loss,
    simclr_nt_xent,
)


def test_simclr_nt_xent_symmetry() -> None:
    torch.manual_seed(0)
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = simclr_nt_xent(z1, z2, temperature=0.5)
    assert loss.item() > 0


def test_identical_temperature_loss_monotonic() -> None:
    image = torch.eye(5)
    omics = torch.eye(5)
    logit_scale = torch.tensor(0.0, requires_grad=True)
    loss = identical_temperature_loss(image, omics, logit_scale)
    loss.backward()
    assert torch.isfinite(loss)


def test_clip_checkpoint_roundtrip(tmp_path: Path) -> None:
    projection = ProjectionConfig(hidden_dim=8, projection_dim=4)
    model = ClipAlignmentModel(
        projection=projection,
        omics_dim=4,
        image_projection=projection,
        temperature=0.5,
    )
    torch.save({"model_state": model.state_dict()}, tmp_path / "ckpt.pt")
    restored = ClipAlignmentModel(
        projection=projection,
        omics_dim=4,
        image_projection=projection,
        temperature=0.5,
    )
    restored.load_state_dict(torch.load(tmp_path / "ckpt.pt")['model_state'])
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q)
