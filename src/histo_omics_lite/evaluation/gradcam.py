"""Minimal Grad-CAM implementation for ResNet-based encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    score: float


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_hook = target_layer.register_forward_hook(self._capture_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, module, inputs, output) -> None:  # type: ignore[override]
        self._activations = output.detach()

    def _capture_gradients(self, module, grad_input, grad_output) -> None:  # type: ignore[override]
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        *,
        image_fn: Callable[[], torch.Tensor],
        score_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> GradCAMResult:
        image = image_fn()
        image.requires_grad_(True)
        score = score_fn(image)
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        pooled_gradients = torch.mean(self._gradients, dim=(0, 2, 3))
        activations = self._activations.clone()
        for idx in range(pooled_gradients.size(0)):
            activations[:, idx, :, :] *= pooled_gradients[idx]

        heatmap = activations.mean(dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return GradCAMResult(heatmap=heatmap.cpu().numpy(), score=float(score.item()))

    def close(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass
