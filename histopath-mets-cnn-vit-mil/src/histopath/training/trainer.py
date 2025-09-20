"""Training utilities and trainer wrapper."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch

from .callbacks import get_callbacks
from ..utils.seed import seed_everything


class HistopathTrainer:
    """Wrapper for PyTorch Lightning trainer with histopathology-specific configurations."""
    
    def __init__(
        self,
        # Training configuration
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        strategy: Optional[str] = None,
        precision: Union[int, str] = 32,
        
        # Logging configuration
        logger_type: str = "mlflow",
        experiment_name: str = "histopath_experiments",
        run_name: Optional[str] = None,
        log_dir: Path = Path("logs"),
        
        # Checkpoint configuration
        checkpoint_dir: Path = Path("checkpoints"),
        monitor_metric: str = "val/auroc",
        save_top_k: int = 3,
        
        # Callback configuration
        patience: int = 15,
        min_delta: float = 0.001,
        unfreeze_epoch: Optional[int] = None,
        log_gradients: bool = False,
        log_attention: bool = False,
        log_model_stats: bool = False,
        
        # Other configuration
        deterministic: bool = True,
        seed: int = 42,
        profiler: Optional[str] = None,
        fast_dev_run: bool = False,
        overfit_batches: Union[int, float] = 0.0,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        check_val_every_n_epoch: int = 1,
        
        **kwargs
    ):
        """
        Initialize histopathology trainer.
        
        Args:
            max_epochs: Maximum number of training epochs
            accelerator: Type of accelerator (auto, cpu, gpu, tpu)
            devices: Number of devices or list of device ids
            strategy: Training strategy (ddp, ddp_spawn, etc.)
            precision: Training precision (16, 32, 64, or "bf16")
            
            logger_type: Type of logger (mlflow, tensorboard, wandb)
            experiment_name: Name of the experiment
            run_name: Name of the run (optional)
            log_dir: Directory for logs
            
            checkpoint_dir: Directory for checkpoints
            monitor_metric: Metric to monitor for checkpointing
            save_top_k: Number of best models to save
            
            patience: Early stopping patience
            min_delta: Minimum change for improvement
            unfreeze_epoch: Epoch to unfreeze backbone
            log_gradients: Whether to log gradient norms
            log_attention: Whether to log attention weights
            log_model_stats: Whether to log model statistics
            
            deterministic: Whether to use deterministic training
            seed: Random seed
            profiler: Profiler to use (simple, advanced, pytorch)
            fast_dev_run: Run single batch for debugging
            overfit_batches: Number of batches to overfit (for debugging)
            limit_train_batches: Limit training batches
            limit_val_batches: Limit validation batches
            limit_test_batches: Limit test batches
            val_check_interval: Validation check interval
            check_val_every_n_epoch: Check validation every N epochs
        """
        
        # Set seed for reproducibility
        if deterministic:
            seed_everything(seed, deterministic=True)
        
        # Store configuration
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.precision = precision
        self.deterministic = deterministic
        self.seed = seed
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger(
            logger_type, experiment_name, run_name, self.log_dir
        )
        
        # Setup callbacks
        self.callbacks = get_callbacks(
            checkpoint_dir=self.checkpoint_dir,
            monitor_metric=monitor_metric,
            patience=patience,
            min_delta=min_delta,
            unfreeze_epoch=unfreeze_epoch,
            log_gradients=log_gradients,
            log_attention=log_attention,
            log_model_stats=log_model_stats,
        )
        
        # Setup strategy for multi-GPU training
        if strategy is None and isinstance(devices, (list, int)) and (
            (isinstance(devices, int) and devices > 1) or 
            (isinstance(devices, list) and len(devices) > 1)
        ):
            if torch.cuda.is_available():
                self.strategy = DDPStrategy(find_unused_parameters=False)
            else:
                self.strategy = "auto"
        else:
            self.strategy = strategy
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=self.strategy,
            precision=precision,
            logger=self.logger,
            callbacks=self.callbacks,
            deterministic=deterministic,
            profiler=profiler,
            fast_dev_run=fast_dev_run,
            overfit_batches=overfit_batches,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            **kwargs
        )
    
    def fit(
        self, 
        model: pl.LightningModule, 
        datamodule: pl.LightningDataModule,
        ckpt_path: Optional[str] = None
    ) -> None:
        """
        Fit the model.
        
        Args:
            model: Lightning module to train
            datamodule: Lightning data module
            ckpt_path: Path to checkpoint to resume from
        """
        self.trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    def test(
        self, 
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = "best",
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Test the model.
        
        Args:
            model: Lightning module to test
            datamodule: Lightning data module
            ckpt_path: Path to checkpoint or "best" for best checkpoint
            verbose: Whether to print results
            
        Returns:
            Test results
        """
        return self.trainer.test(
            model=model, 
            datamodule=datamodule, 
            ckpt_path=ckpt_path,
            verbose=verbose
        )
    
    def validate(
        self, 
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Validate the model.
        
        Args:
            model: Lightning module to validate
            datamodule: Lightning data module
            ckpt_path: Path to checkpoint
            verbose: Whether to print results
            
        Returns:
            Validation results
        """
        return self.trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=verbose
        )
    
    def predict(
        self,
        model: Optional[pl.LightningModule] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        ckpt_path: Optional[str] = "best",
        return_predictions: bool = True
    ) -> List[Any]:
        """
        Generate predictions.
        
        Args:
            model: Lightning module
            datamodule: Lightning data module
            ckpt_path: Path to checkpoint
            return_predictions: Whether to return predictions
            
        Returns:
            Predictions
        """
        return self.trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=return_predictions
        )
    
    def _setup_logger(
        self, 
        logger_type: str, 
        experiment_name: str, 
        run_name: Optional[str],
        log_dir: Path
    ) -> Union[MLFlowLogger, TensorBoardLogger]:
        """Setup logger based on type."""
        
        if logger_type.lower() == "mlflow":
            # MLFlow logger
            return MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                save_dir=str(log_dir),
                log_model=True,
                tags={
                    "framework": "pytorch-lightning",
                    "domain": "histopathology",
                    "task": "classification"
                }
            )
        
        elif logger_type.lower() == "tensorboard":
            # TensorBoard logger
            return TensorBoardLogger(
                save_dir=str(log_dir),
                name=experiment_name,
                version=run_name,
                log_graph=True
            )
        
        else:
            raise ValueError(f"Unsupported logger type: {logger_type}")
    
    @property
    def best_model_path(self) -> Optional[str]:
        """Get path to best checkpoint."""
        checkpoint_callback = None
        for callback in self.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback:
            return checkpoint_callback.best_model_path
        return None
    
    @property
    def best_model_score(self) -> Optional[float]:
        """Get best model score."""
        checkpoint_callback = None
        for callback in self.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback:
            return checkpoint_callback.best_model_score.item()
        return None
    
    def save_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Save hyperparameters to logger."""
        if hasattr(self.logger, 'log_hyperparams'):
            self.logger.log_hyperparams(hyperparameters)


def create_trainer_from_config(config: Dict[str, Any]) -> HistopathTrainer:
    """
    Create trainer from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured trainer
    """
    trainer_config = config.get('trainer', {})
    
    return HistopathTrainer(**trainer_config)


def get_available_gpus() -> int:
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def setup_distributed_training() -> Dict[str, Any]:
    """Setup configuration for distributed training."""
    num_gpus = get_available_gpus()
    
    if num_gpus > 1:
        return {
            "accelerator": "gpu",
            "devices": num_gpus,
            "strategy": DDPStrategy(find_unused_parameters=False),
        }
    elif num_gpus == 1:
        return {
            "accelerator": "gpu",
            "devices": 1,
        }
    else:
        return {
            "accelerator": "cpu",
            "devices": 1,
        }