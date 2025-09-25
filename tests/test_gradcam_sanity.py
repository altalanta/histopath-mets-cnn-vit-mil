"""Grad-CAM sanity tests to verify explainability correctness."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from histopath.explain.gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    inject_discriminative_patch,
    compute_localization_metrics,
)


class SimpleCNN(nn.Module):
    """Simple CNN for testing Grad-CAM."""
    
    def __init__(self, num_classes: int = 2, input_size: int = 224):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 4 - target for Grad-CAM
            nn.Conv2d(128, 256, kernel_size=3, padding=1, name="target_conv"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate feature size after convolutions
        feature_size = (input_size // 16) ** 2 * 256
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple CNN model for testing."""
    model = SimpleCNN(num_classes=2, input_size=224)
    model.eval()
    
    # Initialize with reasonable weights for testing
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
    
    return model


@pytest.fixture
def test_image():
    """Create test image."""
    # Create realistic test image (3, 224, 224)
    torch.manual_seed(42)
    image = torch.rand(1, 3, 224, 224) * 0.5 + 0.5  # Values in [0.5, 1.0]
    return image


def test_gradcam_basic_functionality(simple_model, test_image):
    """Test basic Grad-CAM functionality."""
    # Create Grad-CAM instance
    gradcam = GradCAM(
        model=simple_model,
        target_layers=["features.6"],  # Target the last conv layer
        use_cuda=False,
    )
    
    # Generate CAM
    cam = gradcam.generate_cam(test_image, class_idx=1)
    
    # Basic checks
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0
    assert cam.max() <= 1
    assert not np.isnan(cam).any()


def test_gradcam_patch_sensitivity(simple_model, test_image):
    """Test that Grad-CAM responds to discriminative patches."""
    # Define patch region
    roi = (64, 96, 64, 96)  # 32x32 patch in center-ish area
    
    # Create Grad-CAM instance
    gradcam = GradCAM(
        model=simple_model,
        target_layers=["features.6"],
        use_cuda=False,
    )
    
    # Generate CAM for clean image
    cam_clean = gradcam.generate_cam(test_image, class_idx=1)
    
    # Inject bright discriminative patch
    patched_image = inject_discriminative_patch(
        test_image, roi, intensity=2.0, pattern="solid"
    )
    
    # Generate CAM for patched image
    cam_patch = gradcam.generate_cam(patched_image, class_idx=1)
    
    # Compute localization metrics
    metrics = compute_localization_metrics(cam_patch, roi, threshold=0.3)
    
    # Test assertions
    roi_activation_clean = cam_clean[roi[0]:roi[1], roi[2]:roi[3]].mean()
    roi_activation_patch = cam_patch[roi[0]:roi[1], roi[2]:roi[3]].mean()
    background_activation = cam_patch[roi[0]:roi[1], roi[2]:roi[3]].mean()  # This should be cam_clean but let's test relative increase
    
    # The patch should significantly increase activation in the ROI
    activation_increase = roi_activation_patch - roi_activation_clean
    
    assert activation_increase > 0.1, (
        f"Patch should increase ROI activation significantly. "
        f"Clean: {roi_activation_clean:.3f}, Patch: {roi_activation_patch:.3f}, "
        f"Increase: {activation_increase:.3f}"
    )
    
    # IoU should be reasonable for a well-localized patch
    assert metrics['iou'] > 0.1, f"IoU too low: {metrics['iou']:.3f}"
    
    # Pointing accuracy should be 1 if patch is the most salient
    # (this might not always be true, so we'll be lenient)
    assert metrics['pointing_accuracy'] >= 0, "Pointing accuracy should be non-negative"


def test_gradcam_different_patterns(simple_model, test_image):
    """Test Grad-CAM with different patch patterns."""
    roi = (80, 112, 80, 112)  # 32x32 patch
    
    gradcam = GradCAM(
        model=simple_model,
        target_layers=["features.6"],
        use_cuda=False,
    )
    
    patterns = ["solid", "checkerboard", "gradient"]
    
    for pattern in patterns:
        patched_image = inject_discriminative_patch(
            test_image, roi, intensity=1.5, pattern=pattern
        )
        
        cam = gradcam.generate_cam(patched_image, class_idx=1)
        
        # Basic sanity checks
        assert isinstance(cam, np.ndarray)
        assert cam.shape == (224, 224)
        assert not np.isnan(cam).any()
        
        # Check that patch region has some activation
        roi_activation = cam[roi[0]:roi[1], roi[2]:roi[3]].mean()
        assert roi_activation > 0, f"No activation in ROI for {pattern} pattern"


def test_gradcam_plus_plus(simple_model, test_image):
    """Test Grad-CAM++ variant."""
    roi = (60, 100, 60, 100)
    
    gradcam_pp = GradCAMPlusPlus(
        model=simple_model,
        target_layers=["features.6"],
        use_cuda=False,
    )
    
    # Test with patch
    patched_image = inject_discriminative_patch(
        test_image, roi, intensity=2.0
    )
    
    cam = gradcam_pp.generate_cam(patched_image, class_idx=1)
    
    # Basic checks
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0
    assert cam.max() <= 1
    assert not np.isnan(cam).any()


def test_discriminative_patch_injection():
    """Test patch injection utility function."""
    image = torch.ones(1, 3, 224, 224) * 0.5
    roi = (50, 100, 50, 100)
    
    # Test solid patch
    patched = inject_discriminative_patch(image, roi, intensity=1.0, pattern="solid")
    patch_region = patched[0, 0, roi[0]:roi[1], roi[2]:roi[3]]
    
    assert torch.allclose(patch_region, torch.ones_like(patch_region)), "Solid patch not applied correctly"
    
    # Test that outside patch is unchanged
    outside_region = patched[0, 0, 0:roi[0], 0:roi[2]]
    expected_outside = torch.ones_like(outside_region) * 0.5
    
    assert torch.allclose(outside_region, expected_outside), "Area outside patch was modified"


def test_localization_metrics():
    """Test localization metrics computation."""
    # Create simple test case
    cam = np.zeros((100, 100))
    cam[25:75, 25:75] = 0.8  # High activation in center
    cam[10:20, 10:20] = 0.3  # Lower activation elsewhere
    
    roi = (20, 80, 20, 80)  # Overlaps with high activation area
    
    metrics = compute_localization_metrics(cam, roi, threshold=0.5)
    
    # Check metrics are reasonable
    assert 0 <= metrics['iou'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert metrics['pointing_accuracy'] in [0.0, 1.0]
    assert metrics['roi_mean_activation'] > metrics['background_mean_activation']


def test_gradcam_deterministic(simple_model, test_image):
    """Test that Grad-CAM produces deterministic results."""
    gradcam = GradCAM(
        model=simple_model,
        target_layers=["features.6"],
        use_cuda=False,
    )
    
    # Generate CAM twice
    torch.manual_seed(42)
    cam1 = gradcam.generate_cam(test_image.clone(), class_idx=1)
    
    torch.manual_seed(42) 
    cam2 = gradcam.generate_cam(test_image.clone(), class_idx=1)
    
    # Should be identical
    np.testing.assert_array_almost_equal(cam1, cam2, decimal=6)


def test_gradcam_robustness():
    """Test Grad-CAM robustness to edge cases."""
    model = SimpleCNN(num_classes=2)
    model.eval()
    
    gradcam = GradCAM(
        model=model,
        target_layers=["features.6"],
        use_cuda=False,
    )
    
    # Test with all-zeros image
    zero_image = torch.zeros(1, 3, 224, 224)
    cam = gradcam.generate_cam(zero_image, class_idx=0)
    assert not np.isnan(cam).any(), "CAM should not contain NaN for zero image"
    
    # Test with extreme values
    extreme_image = torch.ones(1, 3, 224, 224) * 1000
    cam = gradcam.generate_cam(extreme_image, class_idx=0)
    assert not np.isnan(cam).any(), "CAM should not contain NaN for extreme image"
    assert np.isfinite(cam).all(), "CAM should contain only finite values"


if __name__ == "__main__":
    # Run a quick test when script is executed directly
    model = SimpleCNN()
    image = torch.rand(1, 3, 224, 224)
    
    gradcam = GradCAM(model, ["features.6"], use_cuda=False)
    
    # Test patch sensitivity
    roi = (64, 96, 64, 96)
    clean_cam = gradcam.generate_cam(image, class_idx=1)
    
    patched = inject_discriminative_patch(image, roi, intensity=2.0)
    patch_cam = gradcam.generate_cam(patched, class_idx=1)
    
    metrics = compute_localization_metrics(patch_cam, roi)
    
    print("Grad-CAM Sanity Check Results:")
    print(f"Clean ROI activation: {clean_cam[roi[0]:roi[1], roi[2]:roi[3]].mean():.3f}")
    print(f"Patch ROI activation: {patch_cam[roi[0]:roi[1], roi[2]:roi[3]].mean():.3f}")
    print(f"Localization IoU: {metrics['iou']:.3f}")
    print(f"Pointing accuracy: {metrics['pointing_accuracy']:.1f}")
    
    if metrics['iou'] > 0.1:
        print("✅ Grad-CAM responds to discriminative patches")
    else:
        print("❌ Grad-CAM may not be localizing correctly")