"""PyTorch Lightning modules for training and evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import numpy as np

from .backbones import create_backbone
from .mil import MILModel


class HistopathClassifier(pl.LightningModule):
    """PyTorch Lightning module for histopathology tile classification."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        class_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize histopathology classifier.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone initially
            optimizer: Optimizer type ("adam", "sgd", "adamw")
            scheduler: Learning rate scheduler ("cosine", "step", "plateau")
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            class_weights: Class weights for imbalanced datasets
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = create_backbone(
            architecture=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
        
        # Loss function
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        if num_classes == 2:
            self.val_auroc = AUROC(task="binary")
            self.test_auroc = AUROC(task="binary")
        else:
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)
        
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store predictions for analysis
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        if self.hparams.num_classes == 2:
            self.val_auroc(probs[:, 1], labels)
        else:
            self.val_auroc(probs, labels)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.test_confusion_matrix(preds, labels)
        
        if self.hparams.num_classes == 2:
            self.test_auroc(probs[:, 1], labels)
        else:
            self.test_auroc(probs, labels)
        
        # Store predictions for analysis
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_probabilities.extend(probs.cpu().numpy())
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Scheduler
        if self.hparams.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.max_epochs // 3,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/auroc",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer


class MILClassifier(pl.LightningModule):
    """PyTorch Lightning module for MIL classification."""
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        mil_type: str = "attention",
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        max_epochs: int = 100,
        class_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize MIL classifier.
        
        Args:
            backbone_name: Name of backbone architecture
            mil_type: Type of MIL aggregation
            feature_dim: Feature dimension
            hidden_dim: Hidden dimension for MIL
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay
            pretrained_backbone: Whether to use pretrained backbone
            freeze_backbone: Whether to freeze backbone initially
            optimizer: Optimizer type
            scheduler: Learning rate scheduler
            max_epochs: Total number of training epochs
            class_weights: Class weights for imbalanced datasets
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = MILModel(
            backbone_name=backbone_name,
            mil_type=mil_type,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
        
        # Loss function
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics (same as HistopathClassifier)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        if num_classes == 2:
            self.val_auroc = AUROC(task="binary")
            self.test_auroc = AUROC(task="binary")
        else:
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
            self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)
        
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store predictions and attention weights
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
        self.test_attention_weights = []
        self.test_bag_ids = []
    
    def forward(
        self, 
        images: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        return self.model(images, return_attention=return_attention)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['images']  # (B, N, C, H, W)
        labels = batch['label']   # (B,)
        
        # Forward pass
        logits, _ = self.forward(images, return_attention=False)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['images']
        labels = batch['label']
        
        # Forward pass
        logits, attention_weights = self.forward(images, return_attention=True)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        if self.hparams.num_classes == 2:
            self.val_auroc(probs[:, 1], labels)
        else:
            self.val_auroc(probs, labels)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        images = batch['images']
        labels = batch['label']
        bag_ids = batch.get('bag_id', [f'bag_{batch_idx}_{i}' for i in range(len(labels))])
        
        # Forward pass
        logits, attention_weights = self.forward(images, return_attention=True)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.test_confusion_matrix(preds, labels)
        
        if self.hparams.num_classes == 2:
            self.test_auroc(probs[:, 1], labels)
        else:
            self.test_auroc(probs, labels)
        
        # Store predictions and attention weights for analysis
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_probabilities.extend(probs.cpu().numpy())
        self.test_bag_ids.extend(bag_ids)
        
        if attention_weights is not None:
            self.test_attention_weights.extend(attention_weights.cpu().numpy())
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Same implementation as HistopathClassifier
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        if self.hparams.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.max_epochs // 3,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/auroc",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Log confusion matrix
        cm = self.test_confusion_matrix.compute()
        self.logger.experiment.log_confusion_matrix(
            matrix=cm.cpu().numpy(),
            labels=list(range(self.hparams.num_classes))
        )
        
        # Save predictions and attention weights for visualization
        if hasattr(self.logger, 'save_dir'):
            save_dir = self.logger.save_dir
            np.save(f"{save_dir}/test_predictions.npy", np.array(self.test_predictions))
            np.save(f"{save_dir}/test_targets.npy", np.array(self.test_targets))
            np.save(f"{save_dir}/test_probabilities.npy", np.array(self.test_probabilities))
            
            if self.test_attention_weights:
                np.save(f"{save_dir}/test_attention_weights.npy", np.array(self.test_attention_weights))
    
    def unfreeze_backbone(self, epoch: int = 10) -> None:
        """Unfreeze backbone after specified epochs."""
        if hasattr(self.model.backbone, 'unfreeze_backbone'):
            self.model.backbone.unfreeze_backbone()
            print(f"Unfroze backbone at epoch {epoch}")