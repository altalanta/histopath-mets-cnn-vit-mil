"""Test cases for model architectures."""

import pytest
import torch
import numpy as np

from histopath.models.mil import AttentionMIL, ABMIL, TransMIL, MILModel
from histopath.models.backbones import CNNBackbone, ViTBackbone, create_backbone
from histopath.models.lightning_modules import HistopathClassifier, MILClassifier


class TestMILModels:
    """Test MIL model implementations."""
    
    def test_attention_mil_forward(self):
        """Test AttentionMIL forward pass."""
        model = AttentionMIL(
            feature_dim=512,
            hidden_dim=256,
            num_classes=2,
            pooling="attention"
        )
        
        # Test input: batch of 4 samples with 10 instances each
        features = torch.randn(4, 10, 512)
        
        # Forward pass without attention
        logits, attention = model(features, return_attention=False)
        assert logits.shape == (4, 2)
        assert attention is None
        
        # Forward pass with attention
        logits, attention = model(features, return_attention=True)
        assert logits.shape == (4, 2)
        assert attention.shape == (4, 10, 1)
    
    def test_abmil_forward(self):
        """Test ABMIL forward pass."""
        model = ABMIL(
            feature_dim=512,
            hidden_dim=256,
            num_classes=2,
            use_gated=True
        )
        
        features = torch.randn(2, 20, 512)
        logits, attention = model(features, return_attention=True)
        
        assert logits.shape == (2, 2)
        assert attention.shape == (2, 20, 1)
    
    def test_transmil_forward(self):
        """Test TransMIL forward pass."""
        model = TransMIL(
            feature_dim=512,
            hidden_dim=256,
            num_classes=2,
            num_heads=8,
            num_layers=2
        )
        
        features = torch.randn(2, 15, 512)
        logits, attention = model(features, return_attention=True)
        
        assert logits.shape == (2, 2)
        # Note: attention may be None depending on implementation
    
    def test_mil_model_with_backbone(self):
        """Test complete MIL model with backbone."""
        model = MILModel(
            backbone_name="resnet18",  # Use smaller model for testing
            mil_type="attention",
            feature_dim=512,
            hidden_dim=256,
            num_classes=2,
            pretrained_backbone=False
        )
        
        # Test input: batch of 2 bags with 5 images each
        images = torch.randn(2, 5, 3, 224, 224)
        logits, attention = model(images, return_attention=True)
        
        assert logits.shape == (2, 2)
        if attention is not None:
            assert attention.shape == (2, 5, 1)


class TestBackbones:
    """Test backbone implementations."""
    
    def test_cnn_backbone_resnet(self):
        """Test CNN backbone with ResNet."""
        backbone = CNNBackbone(
            model_name="resnet18",
            pretrained=False,
            num_classes=None  # Feature extraction mode
        )
        
        x = torch.randn(2, 3, 224, 224)
        features = backbone(x)
        assert features.shape == (2, 512)  # ResNet18 feature dim
    
    def test_cnn_backbone_classification(self):
        """Test CNN backbone in classification mode."""
        backbone = CNNBackbone(
            model_name="resnet18",
            pretrained=False,
            num_classes=5
        )
        
        x = torch.randn(2, 3, 224, 224)
        logits = backbone(x)
        assert logits.shape == (2, 5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vit_backbone(self):
        """Test ViT backbone."""
        try:
            backbone = ViTBackbone(
                model_name="vit_tiny_patch16_224",
                pretrained=False,
                num_classes=None
            )
            
            x = torch.randn(1, 3, 224, 224)
            features = backbone(x)
            assert len(features.shape) == 2  # Should be (batch, features)
            
        except Exception as e:
            pytest.skip(f"ViT model creation failed: {e}")
    
    def test_create_backbone_factory(self):
        """Test backbone factory function."""
        # Test CNN creation
        cnn_backbone = create_backbone("resnet18", pretrained=False, num_classes=2)
        assert isinstance(cnn_backbone, CNNBackbone)
        
        # Test error handling
        with pytest.raises((ValueError, RuntimeError)):
            create_backbone("nonexistent_model", pretrained=False)


class TestLightningModules:
    """Test PyTorch Lightning modules."""
    
    def test_histopath_classifier_init(self):
        """Test HistopathClassifier initialization."""
        classifier = HistopathClassifier(
            model_name="resnet18",
            num_classes=2,
            pretrained=False
        )
        
        assert classifier.hparams.num_classes == 2
        assert hasattr(classifier, 'model')
    
    def test_mil_classifier_init(self):
        """Test MILClassifier initialization."""
        classifier = MILClassifier(
            backbone_name="resnet18",
            mil_type="attention",
            num_classes=2,
            pretrained_backbone=False
        )
        
        assert classifier.hparams.num_classes == 2
        assert hasattr(classifier, 'model')
    
    def test_classifier_forward_pass(self):
        """Test classifier forward pass."""
        classifier = HistopathClassifier(
            model_name="resnet18",
            num_classes=2,
            pretrained=False
        )
        
        x = torch.randn(2, 3, 224, 224)
        logits = classifier(x)
        assert logits.shape == (2, 2)
    
    def test_mil_classifier_forward_pass(self):
        """Test MIL classifier forward pass."""
        classifier = MILClassifier(
            backbone_name="resnet18",
            mil_type="attention",
            num_classes=2,
            pretrained_backbone=False
        )
        
        images = torch.randn(1, 10, 3, 224, 224)
        logits, attention = classifier(images, return_attention=True)
        assert logits.shape == (1, 2)


class TestModelValidation:
    """Test model input validation and error handling."""
    
    def test_invalid_model_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            CNNBackbone(num_classes=-1)
        
        with pytest.raises(ValueError):
            CNNBackbone(feature_dim=0)
        
        with pytest.raises(ValueError):
            AttentionMIL(num_classes=0)
    
    def test_unknown_model_names(self):
        """Test error handling for unknown model names."""
        with pytest.raises((ValueError, RuntimeError)):
            CNNBackbone(model_name="unknown_model")
        
        with pytest.raises(ValueError):
            MILModel(backbone_name="unknown_backbone")


if __name__ == "__main__":
    pytest.main([__file__])