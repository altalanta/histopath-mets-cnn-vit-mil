"""Multiple Instance Learning models with attention mechanisms."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model following Ilse et al. 2018.
    
    Reference:
    Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple 
    instance learning. In International conference on machine learning.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = "attention"  # "attention", "mean", "max"
    ):
        """
        Initialize Attention MIL model.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Pooling method for aggregation
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pooling = pooling
        
        # Attention mechanism
        if pooling == "attention":
            self.attention_layers = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def forward(
        self, 
        features: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Instance features (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, num_instances, _ = features.shape
        
        if self.pooling == "attention":
            # Compute attention weights
            attention_weights = self.attention_layers(features)  # (B, N, 1)
            attention_weights = F.softmax(attention_weights, dim=1)  # (B, N, 1)
            
            # Weighted aggregation
            bag_features = torch.sum(
                attention_weights * features, dim=1
            )  # (B, D)
            
        elif self.pooling == "mean":
            bag_features = torch.mean(features, dim=1)
            attention_weights = None
            
        elif self.pooling == "max":
            bag_features, _ = torch.max(features, dim=1)
            attention_weights = None
            
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Classification
        logits = self.classifier(bag_features)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits, None
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ABMIL(nn.Module):
    """
    Attention-based MIL with gated attention mechanism.
    
    Based on the implementation in the attention-based MIL paper
    with additional gating mechanism for improved performance.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_gated: bool = True
    ):
        """
        Initialize ABMIL model.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention
            num_classes: Number of output classes
            dropout: Dropout rate
            use_gated: Whether to use gated attention
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_gated = use_gated
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        if use_gated:
            # Gated attention
            self.attention_U = nn.Linear(hidden_dim, hidden_dim)
            self.attention_V = nn.Linear(hidden_dim, hidden_dim)
            self.attention_w = nn.Linear(hidden_dim, 1)
        else:
            # Standard attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def forward(
        self, 
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Instance features (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Transform features
        h = self.feature_transform(features)  # (B, N, H)
        
        # Compute attention weights
        if self.use_gated:
            # Gated attention mechanism
            u = self.attention_U(h)  # (B, N, H)
            v = self.attention_V(h)  # (B, N, H)
            attention_weights = self.attention_w(torch.tanh(u) * torch.sigmoid(v))  # (B, N, 1)
        else:
            # Standard attention
            attention_weights = self.attention(h)  # (B, N, 1)
        
        attention_weights = F.softmax(attention_weights, dim=1)  # (B, N, 1)
        
        # Weighted aggregation
        bag_features = torch.sum(attention_weights * h, dim=1)  # (B, H)
        
        # Classification
        logits = self.classifier(bag_features)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits, None
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class TransMIL(nn.Module):
    """
    Transformer-based MIL model.
    
    Uses transformer architecture for modeling relationships
    between instances in a bag.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_instances: int = 1000
    ):
        """
        Initialize TransMIL model.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for transformer
            num_classes: Number of output classes
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_instances: Maximum number of instances (for positional encoding)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_instances = max_instances
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_instances)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def forward(
        self, 
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Instance features (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, num_instances, _ = features.shape
        
        # Project features
        h = self.input_projection(features)  # (B, N, H)
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, H)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, N+1, H)
        
        # Apply transformer
        if return_attention:
            # Need to extract attention weights
            attention_weights = []
            for layer in self.transformer.layers:
                h = layer(h)
                # Note: Getting attention weights from transformer layers 
                # requires modifying the forward pass
            transformer_output = h
        else:
            transformer_output = self.transformer(h)  # (B, N+1, H)
        
        # Use classification token for prediction
        cls_output = transformer_output[:, 0]  # (B, H)
        
        # Classification
        logits = self.classifier(cls_output)
        
        # For attention visualization, we can use the attention to the cls token
        if return_attention:
            # This is a simplified version - in practice you'd need to 
            # modify the transformer to return attention weights
            attention_weights = None
        else:
            attention_weights = None
        
        return logits, attention_weights
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor (B, N, D)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].transpose(0, 1)


class MILModel(nn.Module):
    """
    Complete MIL model with backbone feature extractor.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        mil_type: str = "attention",
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        **kwargs
    ):
        """
        Initialize complete MIL model.
        
        Args:
            backbone_name: Name of backbone architecture
            mil_type: Type of MIL aggregation ("attention", "abmil", "transmil")
            feature_dim: Feature dimension after backbone
            hidden_dim: Hidden dimension for MIL layers
            num_classes: Number of output classes
            pretrained_backbone: Whether to use pretrained backbone
            freeze_backbone: Whether to freeze backbone weights
            **kwargs: Additional arguments for MIL model
        """
        super().__init__()
        
        # Feature extractor backbone
        self.backbone = create_backbone(
            architecture=backbone_name,
            pretrained=pretrained_backbone,
            num_classes=None,  # No classification head
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        
        # MIL aggregation model
        if mil_type == "attention":
            self.mil_model = AttentionMIL(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif mil_type == "abmil":
            self.mil_model = ABMIL(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif mil_type == "transmil":
            self.mil_model = TransMIL(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown MIL type: {mil_type}")
    
    def forward(
        self, 
        images: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            images: Bag of images (B, N, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size, num_instances = images.shape[:2]
        
        # Reshape for backbone processing
        images = images.view(-1, *images.shape[2:])  # (B*N, C, H, W)
        
        # Extract features
        features = self.backbone(images)  # (B*N, D)
        
        # Reshape back to bags
        features = features.view(batch_size, num_instances, -1)  # (B, N, D)
        
        # MIL aggregation and classification
        return self.mil_model(features, return_attention=return_attention)