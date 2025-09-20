"""Backbone architectures for feature extraction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm
from torchvision import models


class CNNBackbone(nn.Module):
    """CNN backbone for feature extraction."""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        freeze_backbone: bool = False,
        feature_dim: Optional[int] = None
    ):
        """
        Initialize CNN backbone.
        
        Args:
            model_name: Name of the CNN model (resnet50, efficientnet_b0, etc.)
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (None for feature extraction)
            freeze_backbone: Whether to freeze backbone weights
            feature_dim: Dimension of output features (None to use model default)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model using timm for better model zoo
        if model_name.startswith('resnet'):
            # Use torchvision for standard ResNets
            self.backbone = getattr(models, model_name)(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            
            if num_classes is None:
                # Remove classifier for feature extraction
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                self.backbone.add_module('flatten', nn.Flatten())
            else:
                # Replace classifier
                self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
                
        else:
            # Use timm for other models
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes if num_classes is not None else 0
            )
            self.feature_dim = self.backbone.num_features
        
        # Add custom feature projection if specified
        if feature_dim is not None and feature_dim != self.feature_dim:
            if num_classes is None:
                self.backbone.add_module(
                    'feature_projection',
                    nn.Linear(self.feature_dim, feature_dim)
                )
            self.feature_dim = feature_dim
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Features or logits depending on configuration
        """
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim


class ViTBackbone(nn.Module):
    """Vision Transformer backbone for feature extraction."""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        freeze_backbone: bool = False,
        feature_dim: Optional[int] = None,
        patch_size: int = 16,
        img_size: int = 224
    ):
        """
        Initialize ViT backbone.
        
        Args:
            model_name: Name of the ViT model
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (None for feature extraction)
            freeze_backbone: Whether to freeze backbone weights
            feature_dim: Dimension of output features
            patch_size: Patch size for ViT
            img_size: Input image size
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Load ViT model using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes if num_classes is not None else 0,
            img_size=img_size
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        else:
            # Default for most ViT models
            if 'base' in model_name:
                self.feature_dim = 768
            elif 'large' in model_name:
                self.feature_dim = 1024
            elif 'small' in model_name:
                self.feature_dim = 384
            else:
                self.feature_dim = 768
        
        # Add custom feature projection if specified
        if feature_dim is not None and feature_dim != self.feature_dim:
            self.feature_projection = nn.Linear(self.feature_dim, feature_dim)
            self.feature_dim = feature_dim
        else:
            self.feature_projection = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Features or logits depending on configuration
        """
        features = self.backbone(x)
        features = self.feature_projection(features)
        return features
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both patch and class tokens.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (class_token, patch_tokens)
        """
        # Get features from all layers
        x = self.backbone.patch_embed(x)
        
        # Add position embeddings
        if hasattr(self.backbone, 'pos_embed'):
            x = x + self.backbone.pos_embed
        
        # Add class token
        if hasattr(self.backbone, 'cls_token'):
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        # Apply transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)
        
        # Apply final layer norm
        x = self.backbone.norm(x)
        
        # Split class and patch tokens
        if hasattr(self.backbone, 'cls_token'):
            class_token = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            class_token = x.mean(dim=1)  # Global average pooling
            patch_tokens = x
        
        return class_token, patch_tokens
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of attention maps from each layer
        """
        attention_maps = []
        
        # Patch embedding
        x = self.backbone.patch_embed(x)
        
        # Add position embeddings
        if hasattr(self.backbone, 'pos_embed'):
            x = x + self.backbone.pos_embed
        
        # Add class token
        if hasattr(self.backbone, 'cls_token'):
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        # Apply transformer blocks and collect attention
        for blk in self.backbone.blocks:
            # Get attention weights from the block
            attn_weights = []
            
            def hook_fn(module, input, output):
                if hasattr(output, 'shape') and len(output.shape) == 4:
                    # This is likely attention weights (B, H, N, N)
                    attn_weights.append(output)
            
            # Register hook on attention module
            if hasattr(blk, 'attn'):
                handle = blk.attn.register_forward_hook(hook_fn)
                x = blk(x)
                handle.remove()
                
                if attn_weights:
                    attention_maps.append(attn_weights[0])
            else:
                x = blk(x)
        
        return attention_maps
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim


def create_backbone(
    architecture: str = "resnet50",
    pretrained: bool = True,
    num_classes: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create backbone models.
    
    Args:
        architecture: Model architecture name
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        Backbone model
    """
    if architecture.startswith('vit') or 'transformer' in architecture.lower():
        return ViTBackbone(
            model_name=architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
    else:
        return CNNBackbone(
            model_name=architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )