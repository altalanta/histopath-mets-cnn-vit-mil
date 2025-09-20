"""Evaluation and interpretability modules."""

from .metrics import HistopathMetrics, MILMetrics, bootstrap_ci
from .interpretability import GradCAM, AttentionVisualizer, SHAPExplainer
from .evaluation import ModelEvaluator

__all__ = [
    "HistopathMetrics",
    "MILMetrics", 
    "bootstrap_ci",
    "GradCAM",
    "AttentionVisualizer",
    "SHAPExplainer",
    "ModelEvaluator",
]