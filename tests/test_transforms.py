"""Test cases for data transforms and stain normalization."""

import pytest
import numpy as np
import albumentations as A

from histopath.data.transforms import (
    HistopathTransforms,
    StainNormalizer,
    denormalize_tensor,
)


class TestHistopathTransforms:
    """Test histopathology transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        transform = HistopathTransforms.get_train_transforms(
            size=224, normalize=True, augment_prob=0.5
        )
        
        assert isinstance(transform, A.Compose)
        
        # Test with dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = transform(image=image)
        
        assert "image" in transformed
        assert transformed["image"].shape == (3, 224, 224)  # CHW format after ToTensor

    def test_val_transforms(self):
        """Test validation transforms."""
        transform = HistopathTransforms.get_val_transforms(size=224, normalize=True)
        
        assert isinstance(transform, A.Compose)
        
        # Test with dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = transform(image=image)
        
        assert transformed["image"].shape == (3, 224, 224)

    def test_test_time_augmentation(self):
        """Test test-time augmentation transforms."""
        transforms = HistopathTransforms.get_test_time_augmentation()
        
        assert len(transforms) == 4  # Original + 3 augmentations
        
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        for transform in transforms:
            transformed = transform(image=image)
            assert transformed["image"].shape == (3, 224, 224)

    def test_stain_normalization_transform(self):
        """Test stain normalization transform creation."""
        transform = HistopathTransforms.get_stain_normalization_transform()
        
        assert isinstance(transform, A.Compose)
        
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transformed = transform(image=image)
        
        assert transformed["image"].shape == (3, 256, 256)


class TestStainNormalizer:
    """Test stain normalization implementation."""

    def test_stain_normalizer_init(self):
        """Test StainNormalizer initialization."""
        # Test valid methods
        for method in ["macenko", "vahadane", "reinhard"]:
            normalizer = StainNormalizer(method=method)
            assert normalizer.method == method

        # Test invalid method
        with pytest.raises(ValueError):
            StainNormalizer(method="invalid_method")

    def test_reinhard_normalization(self):
        """Test Reinhard color normalization."""
        normalizer = StainNormalizer(method="reinhard")
        
        # Create dummy reference images
        ref_images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Fit normalizer
        normalizer.fit(ref_images)
        assert normalizer._target_stats is not None
        
        # Transform test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        normalized = normalizer.transform(test_image)
        
        assert normalized.shape == test_image.shape
        assert normalized.dtype == np.uint8

    def test_macenko_normalization(self):
        """Test Macenko stain normalization."""
        normalizer = StainNormalizer(method="macenko")
        
        # Create dummy H&E-like images
        ref_images = [
            np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        # Fit normalizer
        normalizer.fit(ref_images)
        assert normalizer._target_stats is not None
        
        # Transform test image
        test_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        normalized = normalizer.transform(test_image)
        
        assert normalized.shape == test_image.shape
        assert normalized.dtype == np.uint8

    def test_fit_transform(self):
        """Test fit_transform method."""
        normalizer = StainNormalizer(method="reinhard")
        
        images = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        normalized_images = normalizer.fit_transform(images)
        
        assert len(normalized_images) == len(images)
        for norm_img, orig_img in zip(normalized_images, images):
            assert norm_img.shape == orig_img.shape

    def test_empty_images_error(self):
        """Test error handling for empty image list."""
        normalizer = StainNormalizer()
        
        with pytest.raises(ValueError):
            normalizer.fit([])

    def test_robust_error_handling(self):
        """Test robust error handling in transform."""
        normalizer = StainNormalizer(method="reinhard")
        
        # Test transform without fitting (should return original)
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = normalizer.transform(test_image)
        np.testing.assert_array_equal(result, test_image)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_denormalize_tensor(self):
        """Test tensor denormalization."""
        # Create normalized tensor
        tensor = np.random.randn(3, 224, 224).astype(np.float32)
        
        # Denormalize
        denormalized = denormalize_tensor(tensor)
        
        assert denormalized.shape == (224, 224, 3)  # HWC format
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255

    def test_denormalize_tensor_custom_stats(self):
        """Test tensor denormalization with custom statistics."""
        tensor = np.random.randn(3, 128, 128).astype(np.float32)
        
        mean = (0.5, 0.5, 0.5)
        std = (0.2, 0.2, 0.2)
        
        denormalized = denormalize_tensor(tensor, mean=mean, std=std)
        
        assert denormalized.shape == (128, 128, 3)
        assert denormalized.dtype == np.uint8


class TestColorSpaceConversions:
    """Test color space conversion utilities."""

    def test_rgb_to_lab_conversion(self):
        """Test RGB to LAB conversion in StainNormalizer."""
        normalizer = StainNormalizer()
        
        # Test with known RGB values
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 255  # Pure red
        
        lab_image = normalizer._rgb_to_lab(rgb_image)
        
        assert lab_image.shape == rgb_image.shape
        assert not np.allclose(lab_image, rgb_image)  # Should be different

    def test_stain_matrix_computation(self):
        """Test stain matrix computation."""
        normalizer = StainNormalizer(method="macenko")
        
        # Create H&E-like image
        he_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        normalizer._compute_stain_matrix(he_image)
        
        assert normalizer._target_stats is not None
        assert "he_matrix" in normalizer._target_stats
        assert "max_conc" in normalizer._target_stats


if __name__ == "__main__":
    pytest.main([__file__])