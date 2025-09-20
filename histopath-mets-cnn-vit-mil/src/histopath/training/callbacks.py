"""Custom callbacks for training."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import torch
import numpy as np


class UnfreezeBackboneCallback(pl.Callback):
    """Callback to unfreeze backbone after specified epochs."""
    
    def __init__(self, unfreeze_epoch: int = 10):
        """
        Initialize callback.
        
        Args:
            unfreeze_epoch: Epoch at which to unfreeze backbone
        """
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the start of each training epoch."""
        if trainer.current_epoch == self.unfreeze_epoch and not self.unfrozen:
            if hasattr(pl_module, 'unfreeze_backbone'):
                pl_module.unfreeze_backbone(trainer.current_epoch)
                self.unfrozen = True
                trainer.logger.log_metrics(
                    {"backbone_unfrozen": 1}, 
                    step=trainer.global_step
                )


class GradientLoggingCallback(pl.Callback):
    """Callback to log gradient norms."""
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize callback.
        
        Args:
            log_frequency: Frequency of logging (in steps)
        """
        self.log_frequency = log_frequency
    
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called after backward pass."""
        if trainer.global_step % self.log_frequency == 0:
            # Calculate gradient norms
            total_norm = 0.0
            param_count = 0
            
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1. / 2)
            
            # Log gradient norm
            trainer.logger.log_metrics(
                {
                    "grad_norm": total_norm,
                    "grad_norm_log": np.log10(total_norm + 1e-8)
                },
                step=trainer.global_step
            )


class AttentionLoggingCallback(pl.Callback):
    """Callback to log attention weights during training."""
    
    def __init__(self, log_frequency: int = 500, num_samples: int = 4):
        """
        Initialize callback.
        
        Args:
            log_frequency: Frequency of logging (in steps)  
            num_samples: Number of samples to log
        """
        self.log_frequency = log_frequency
        self.num_samples = num_samples
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Dict[str, Any], 
        batch_idx: int
    ) -> None:
        """Called at the end of training batch."""
        if trainer.global_step % self.log_frequency == 0:
            # Only for MIL models
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'mil_model'):
                self._log_attention_weights(trainer, pl_module, batch)
    
    def _log_attention_weights(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        batch: Dict[str, Any]
    ) -> None:
        """Log attention weights."""
        with torch.no_grad():
            pl_module.eval()
            
            # Get a subset of the batch
            images = batch['images'][:self.num_samples]  # (N, T, C, H, W)
            
            # Forward pass to get attention weights
            _, attention_weights = pl_module.forward(images, return_attention=True)
            
            if attention_weights is not None:
                # Log attention statistics
                attn_mean = attention_weights.mean().item()
                attn_std = attention_weights.std().item()
                attn_max = attention_weights.max().item()
                attn_min = attention_weights.min().item()
                
                trainer.logger.log_metrics({
                    "attention/mean": attn_mean,
                    "attention/std": attn_std,
                    "attention/max": attn_max,
                    "attention/min": attn_min,
                }, step=trainer.global_step)
                
                # Log attention distribution (entropy)
                attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1)
                trainer.logger.log_metrics({
                    "attention/entropy_mean": attn_entropy.mean().item(),
                    "attention/entropy_std": attn_entropy.std().item(),
                }, step=trainer.global_step)
            
            pl_module.train()


class ModelStatisticsCallback(pl.Callback):
    """Callback to log model statistics."""
    
    def __init__(self, log_frequency: int = 1000):
        """
        Initialize callback.
        
        Args:
            log_frequency: Frequency of logging (in steps)
        """
        self.log_frequency = log_frequency
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Dict[str, Any], 
        batch_idx: int
    ) -> None:
        """Called at the end of training batch."""
        if trainer.global_step % self.log_frequency == 0:
            self._log_model_stats(trainer, pl_module)
    
    def _log_model_stats(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log model statistics."""
        # Parameter statistics
        param_stats = {}
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param_data = param.data
                param_stats[f"param/{name}/mean"] = param_data.mean().item()
                param_stats[f"param/{name}/std"] = param_data.std().item()
                param_stats[f"param/{name}/norm"] = param_data.norm().item()
        
        # Log a subset to avoid overwhelming the logger
        important_params = [k for k in param_stats.keys() if any(
            layer in k for layer in ['classifier', 'attention', 'fc', 'head']
        )]
        
        filtered_stats = {k: param_stats[k] for k in important_params[:10]}  # Log top 10
        trainer.logger.log_metrics(filtered_stats, step=trainer.global_step)


def get_callbacks(
    checkpoint_dir: Path,
    monitor_metric: str = "val/auroc",
    patience: int = 15,
    min_delta: float = 0.001,
    unfreeze_epoch: Optional[int] = None,
    log_gradients: bool = False,
    log_attention: bool = False,
    log_model_stats: bool = False,
) -> List[pl.Callback]:
    """
    Get list of callbacks for training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor_metric: Metric to monitor for checkpointing
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as improvement
        unfreeze_epoch: Epoch to unfreeze backbone (None to disable)
        log_gradients: Whether to log gradient norms
        log_attention: Whether to log attention weights
        log_model_stats: Whether to log model statistics
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_auroc:.3f}",
        monitor=monitor_metric,
        mode="max" if "auroc" in monitor_metric or "acc" in monitor_metric else "min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="max" if "auroc" in monitor_metric or "acc" in monitor_metric else "min",
        patience=patience,
        min_delta=min_delta,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Progress bar with custom theme
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    callbacks.append(progress_bar)
    
    # Model summary
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    # Unfreeze backbone callback
    if unfreeze_epoch is not None:
        unfreeze_callback = UnfreezeBackboneCallback(unfreeze_epoch=unfreeze_epoch)
        callbacks.append(unfreeze_callback)
    
    # Gradient logging
    if log_gradients:
        grad_callback = GradientLoggingCallback(log_frequency=100)
        callbacks.append(grad_callback)
    
    # Attention logging (for MIL models)
    if log_attention:
        attention_callback = AttentionLoggingCallback(log_frequency=500)
        callbacks.append(attention_callback)
    
    # Model statistics logging
    if log_model_stats:
        stats_callback = ModelStatisticsCallback(log_frequency=1000)
        callbacks.append(stats_callback)
    
    return callbacks


def get_checkpoint_callback(
    checkpoint_dir: Path,
    monitor_metric: str = "val/auroc",
    filename_template: str = "best-{epoch:02d}-{val_auroc:.3f}",
    save_top_k: int = 3
) -> ModelCheckpoint:
    """
    Get model checkpoint callback.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor_metric: Metric to monitor
        filename_template: Template for checkpoint filenames
        save_top_k: Number of best models to save
        
    Returns:
        ModelCheckpoint callback
    """
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_template,
        monitor=monitor_metric,
        mode="max" if "auroc" in monitor_metric or "acc" in monitor_metric else "min",
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )