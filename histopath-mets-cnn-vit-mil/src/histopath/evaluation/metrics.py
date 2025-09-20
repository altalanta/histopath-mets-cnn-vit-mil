"""Comprehensive metrics for histopathology model evaluation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metric: str = "auroc",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUROC, AUPRC)
        metric: Metric to calculate ("auroc", "accuracy", "f1", etc.)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI
        random_state: Random seed
        
    Returns:
        Tuple of (metric_value, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if y_proba is not None:
            y_proba_boot = y_proba[indices]
        else:
            y_proba_boot = None
        
        # Calculate metric
        if metric == "auroc":
            if y_proba_boot is not None:
                score = roc_auc_score(y_true_boot, y_proba_boot)
            else:
                continue
        elif metric == "auprc":
            if y_proba_boot is not None:
                score = average_precision_score(y_true_boot, y_proba_boot)
            else:
                continue
        elif metric == "accuracy":
            score = accuracy_score(y_true_boot, y_pred_boot)
        elif metric == "f1":
            score = f1_score(y_true_boot, y_pred_boot, average='macro')
        elif metric == "precision":
            score = precision_score(y_true_boot, y_pred_boot, average='macro')
        elif metric == "recall":
            score = recall_score(y_true_boot, y_pred_boot, average='macro')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate original metric
    if metric == "auroc":
        if y_proba is not None:
            original_score = roc_auc_score(y_true, y_proba)
        else:
            original_score = 0.5
    elif metric == "auprc":
        if y_proba is not None:
            original_score = average_precision_score(y_true, y_proba)
        else:
            original_score = np.mean(y_true)
    elif metric == "accuracy":
        original_score = accuracy_score(y_true, y_pred)
    elif metric == "f1":
        original_score = f1_score(y_true, y_pred, average='macro')
    elif metric == "precision":
        original_score = precision_score(y_true, y_pred, average='macro')
    elif metric == "recall":
        original_score = recall_score(y_true, y_pred, average='macro')
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return original_score, ci_lower, ci_upper


class HistopathMetrics:
    """Comprehensive metrics for histopathology classification."""
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        bootstrap_ci_metrics: bool = True,
        n_bootstrap: int = 1000
    ) -> Dict[str, Union[float, Tuple[float, float, float]]]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            bootstrap_ci_metrics: Whether to calculate bootstrap CIs
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Probability-based metrics
        if y_proba is not None:
            if self.num_classes == 2:
                # Binary classification
                y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics['auroc'] = roc_auc_score(y_true, y_proba_pos)
                metrics['auprc'] = average_precision_score(y_true, y_proba_pos)
                metrics['brier_score'] = brier_score_loss(y_true, y_proba_pos)
            else:
                # Multi-class classification
                metrics['auroc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['auprc'] = average_precision_score(y_true, y_proba, average='macro')
        
        # Bootstrap confidence intervals
        if bootstrap_ci_metrics and y_proba is not None:
            metrics_for_ci = ['accuracy', 'f1_macro', 'auroc', 'auprc']
            
            for metric_name in metrics_for_ci:
                if metric_name in ['auroc', 'auprc']:
                    y_proba_ci = y_proba[:, 1] if self.num_classes == 2 and y_proba.ndim > 1 else y_proba
                    score, ci_lower, ci_upper = bootstrap_ci(
                        y_true, y_pred, y_proba_ci, metric_name, n_bootstrap
                    )
                else:
                    score, ci_lower, ci_upper = bootstrap_ci(
                        y_true, y_pred, None, metric_name.replace('_macro', ''), n_bootstrap
                    )
                
                metrics[f'{metric_name}_ci'] = (score, ci_lower, ci_upper)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: str = 'true',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', None)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix ({normalize} normalized)')
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            auc = roc_auc_score(y_true, y_proba_pos)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            
        else:
            # Multi-class classification
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                auc = roc_auc_score(y_true_binary, y_proba[:, i])
                
                ax.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
            auc = average_precision_score(y_true, y_proba_pos)
            
            ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc:.3f})')
            
        else:
            # Multi-class classification
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
                auc = average_precision_score(y_true_binary, y_proba[:, i])
                
                ax.plot(recall, precision, linewidth=2, label=f'{class_name} (AUC = {auc:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba_pos, n_bins=n_bins
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2, label='Model')
            
        else:
            # Multi-class - plot for each class
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_proba[:, i], n_bins=n_bins
                )
                
                ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                       linewidth=2, label=class_name)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class MILMetrics(HistopathMetrics):
    """Metrics specific to Multiple Instance Learning."""
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """Initialize MIL metrics."""
        super().__init__(num_classes, class_names)
    
    def calculate_bag_level_metrics(
        self,
        y_true_bags: np.ndarray,
        y_pred_bags: np.ndarray,
        y_proba_bags: Optional[np.ndarray] = None,
        attention_weights: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Union[float, Tuple[float, float, float]]]:
        """
        Calculate bag-level metrics for MIL.
        
        Args:
            y_true_bags: True bag labels
            y_pred_bags: Predicted bag labels
            y_proba_bags: Predicted bag probabilities
            attention_weights: List of attention weights for each bag
            **kwargs: Additional arguments for parent method
            
        Returns:
            Dictionary of metrics including attention statistics
        """
        # Get standard metrics
        metrics = self.calculate_metrics(
            y_true_bags, y_pred_bags, y_proba_bags, **kwargs
        )
        
        # Add attention-specific metrics
        if attention_weights is not None:
            attention_stats = self._analyze_attention_weights(attention_weights)
            metrics.update(attention_stats)
        
        return metrics
    
    def _analyze_attention_weights(
        self, 
        attention_weights: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Analyze attention weight statistics.
        
        Args:
            attention_weights: List of attention weights for each bag
            
        Returns:
            Dictionary of attention statistics
        """
        stats = {}
        
        # Collect all attention weights
        all_weights = np.concatenate([w.flatten() for w in attention_weights])
        
        # Basic statistics
        stats['attention_mean'] = np.mean(all_weights)
        stats['attention_std'] = np.std(all_weights)
        stats['attention_min'] = np.min(all_weights)
        stats['attention_max'] = np.max(all_weights)
        
        # Sparsity measures
        stats['attention_sparsity'] = np.sum(all_weights < 0.01) / len(all_weights)
        stats['attention_gini'] = self._gini_coefficient(all_weights)
        
        # Entropy (measure of attention distribution)
        entropies = []
        for weights in attention_weights:
            weights_flat = weights.flatten()
            # Add small epsilon to avoid log(0)
            entropy = -np.sum(weights_flat * np.log(weights_flat + 1e-8))
            entropies.append(entropy)
        
        stats['attention_entropy_mean'] = np.mean(entropies)
        stats['attention_entropy_std'] = np.std(entropies)
        
        return stats
    
    def _gini_coefficient(self, x: np.ndarray) -> float:
        """Calculate Gini coefficient for attention weights."""
        # Sort values
        sorted_x = np.sort(x)
        n = len(sorted_x)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_x)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_x))) / (n * cumsum[-1]) - (n + 1) / n
        
        return gini
    
    def plot_attention_distribution(
        self,
        attention_weights: List[np.ndarray],
        y_true: np.ndarray,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """
        Plot attention weight distributions by class.
        
        Args:
            attention_weights: List of attention weights
            y_true: True labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Separate by class
        class_weights = {i: [] for i in range(self.num_classes)}
        for weights, label in zip(attention_weights, y_true):
            class_weights[label].extend(weights.flatten())
        
        # Plot 1: Histograms by class
        for i, class_name in enumerate(self.class_names):
            if class_weights[i]:
                axes[0].hist(class_weights[i], alpha=0.7, label=class_name, bins=50)
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attention Weight Distribution')
        axes[0].legend()
        
        # Plot 2: Box plots by class
        box_data = [class_weights[i] for i in range(self.num_classes) if class_weights[i]]
        box_labels = [self.class_names[i] for i in range(self.num_classes) if class_weights[i]]
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Attention Weight Box Plots')
        
        # Plot 3: Attention entropy by class
        entropies_by_class = {i: [] for i in range(self.num_classes)}
        for weights, label in zip(attention_weights, y_true):
            weights_flat = weights.flatten()
            entropy = -np.sum(weights_flat * np.log(weights_flat + 1e-8))
            entropies_by_class[label].append(entropy)
        
        entropy_data = [entropies_by_class[i] for i in range(self.num_classes) if entropies_by_class[i]]
        axes[2].boxplot(entropy_data, labels=box_labels)
        axes[2].set_ylabel('Attention Entropy')
        axes[2].set_title('Attention Entropy by Class')
        
        plt.tight_layout()
        return fig