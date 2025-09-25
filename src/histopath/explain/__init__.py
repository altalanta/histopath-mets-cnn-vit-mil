"""Explainability module for histopathology models."""

from .gradcam import GradCAM, GradCAMPlusPlus, inject_discriminative_patch, compute_localization_metrics

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus", 
    "inject_discriminative_patch",
    "compute_localization_metrics",
]