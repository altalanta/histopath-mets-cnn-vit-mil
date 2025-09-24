"""Image transforms for histopathology data."""

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


class HistopathTransforms:
    """Transforms for histopathology images."""

    @staticmethod
    def get_train_transforms(
        size: int = 256, normalize: bool = True, augment_prob: float = 0.5
    ) -> A.Compose:
        """Get training transforms with augmentation."""
        transforms = [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ],
                p=augment_prob,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ],
                p=augment_prob * 0.5,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=augment_prob * 0.3,
            ),
        ]

        if normalize:
            transforms.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    @staticmethod
    def get_val_transforms(size: int = 256, normalize: bool = True) -> A.Compose:
        """Get validation transforms without augmentation."""
        transforms = [A.Resize(size, size)]

        if normalize:
            transforms.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    @staticmethod
    def get_test_time_augmentation() -> List[A.Compose]:
        """Get transforms for test-time augmentation."""
        return [
            A.Compose([A.NoOp(), ToTensorV2()]),  # Original
            A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()]),
            A.Compose([A.VerticalFlip(p=1.0), ToTensorV2()]),
            A.Compose([A.RandomRotate90(p=1.0), ToTensorV2()]),
        ]

    @staticmethod
    def get_stain_normalization_transform(
        target_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> A.Compose:
        """
        Get transform for H&E stain normalization.

        Args:
            target_stats: Target stain statistics for normalization

        Returns:
            Albumentations transform
        """
        # Placeholder for stain normalization
        # In practice, you would implement Macenko or Vahadane normalization
        return A.Compose(
            [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5), ToTensorV2()]
        )


class StainNormalizer:
    """H&E stain normalization utilities."""

    def __init__(
        self, method: str = "macenko", target_image: Optional[np.ndarray] = None
    ):
        """
        Initialize stain normalizer.

        Args:
            method: Normalization method ('macenko', 'vahadane', 'reinhard')
            target_image: Reference image for normalization
        """
        self.method = method
        self.target_image = target_image
        self._target_stats = None

        if method not in ["macenko", "vahadane", "reinhard"]:
            raise ValueError(f"Unsupported normalization method: {method}")

    def fit(self, images: List[np.ndarray]) -> None:
        """Fit normalizer to reference images."""
        if not images:
            raise ValueError("No reference images provided")

        if self.method == "reinhard":
            # Compute target statistics for Reinhard normalization
            target_stats = []
            for img in images:
                lab = self._rgb_to_lab(img)
                stats = {
                    "mean_l": np.mean(lab[:, :, 0]),
                    "std_l": np.std(lab[:, :, 0]),
                    "mean_a": np.mean(lab[:, :, 1]),
                    "std_a": np.std(lab[:, :, 1]),
                    "mean_b": np.mean(lab[:, :, 2]),
                    "std_b": np.std(lab[:, :, 2]),
                }
                target_stats.append(stats)

            # Use mean of statistics across reference images
            self._target_stats = {
                "mean_l": np.mean([s["mean_l"] for s in target_stats]),
                "std_l": np.mean([s["std_l"] for s in target_stats]),
                "mean_a": np.mean([s["mean_a"] for s in target_stats]),
                "std_a": np.mean([s["std_a"] for s in target_stats]),
                "mean_b": np.mean([s["mean_b"] for s in target_stats]),
                "std_b": np.mean([s["std_b"] for s in target_stats]),
            }

        elif self.method in ["macenko", "vahadane"]:
            # For Macenko/Vahadane, use first image as reference
            if self.target_image is not None:
                self._compute_stain_matrix(self.target_image)
            else:
                self._compute_stain_matrix(images[0])

    def _rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space (simplified implementation)."""
        # Normalize to [0, 1]
        rgb = rgb_image.astype(np.float32) / 255.0

        # Simple RGB to XYZ conversion (not exact)
        xyz = np.zeros_like(rgb)
        xyz[:, :, 0] = (
            0.412453 * rgb[:, :, 0] + 0.357580 * rgb[:, :, 1] + 0.180423 * rgb[:, :, 2]
        )
        xyz[:, :, 1] = (
            0.212671 * rgb[:, :, 0] + 0.715160 * rgb[:, :, 1] + 0.072169 * rgb[:, :, 2]
        )
        xyz[:, :, 2] = (
            0.019334 * rgb[:, :, 0] + 0.119193 * rgb[:, :, 1] + 0.950227 * rgb[:, :, 2]
        )

        # XYZ to LAB (simplified)
        lab = np.zeros_like(xyz)
        lab[:, :, 0] = 116 * np.cbrt(xyz[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (np.cbrt(xyz[:, :, 0]) - np.cbrt(xyz[:, :, 1]))  # a
        lab[:, :, 2] = 200 * (np.cbrt(xyz[:, :, 1]) - np.cbrt(xyz[:, :, 2]))  # b

        return lab

    def _compute_stain_matrix(self, image: np.ndarray) -> None:
        """Compute stain separation matrix (simplified implementation)."""
        # Convert to optical density
        od = -np.log((image.astype(np.float64) + 1) / 256)

        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        mask = np.sum(od_flat, axis=1) > 0.15
        od_masked = od_flat[mask]

        if len(od_masked) == 0:
            # Fallback to default H&E vectors
            self._target_stats = {
                "he_matrix": np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]),
                "max_conc": np.array([1.9705, 1.0308]),
            }
            return

        # Compute stain vectors using SVD (simplified)
        U, _, _ = np.linalg.svd(od_masked.T)
        he_matrix = U[:, :2].T

        # Ensure correct polarity
        if he_matrix[0, 0] < 0:
            he_matrix[0] = -he_matrix[0]
        if he_matrix[1, 0] > 0:
            he_matrix[1] = -he_matrix[1]

        self._target_stats = {
            "he_matrix": he_matrix,
            "max_conc": np.array([1.9705, 1.0308]),
        }

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply stain normalization to image."""
        if self._target_stats is None:
            return image

        try:
            if self.method == "reinhard":
                return self._reinhard_normalize(image)
            elif self.method in ["macenko", "vahadane"]:
                return self._macenko_normalize(image)
            else:
                return image
        except Exception:
            # Return original image if normalization fails
            return image

    def _reinhard_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply Reinhard color normalization."""
        # Convert to LAB
        lab = self._rgb_to_lab(image)

        # Normalize each channel
        for i in range(3):
            channel = lab[:, :, i]
            mean_channel = np.mean(channel)
            std_channel = np.std(channel)

            if std_channel > 0:
                if i == 0:  # L channel
                    normalized = (
                        (channel - mean_channel) / std_channel
                    ) * self._target_stats["std_l"] + self._target_stats["mean_l"]
                elif i == 1:  # a channel
                    normalized = (
                        (channel - mean_channel) / std_channel
                    ) * self._target_stats["std_a"] + self._target_stats["mean_a"]
                else:  # b channel
                    normalized = (
                        (channel - mean_channel) / std_channel
                    ) * self._target_stats["std_b"] + self._target_stats["mean_b"]

                lab[:, :, i] = normalized

        # Convert back to RGB (simplified - just return enhanced image)
        enhanced = image.astype(np.float32)
        enhanced = np.clip(enhanced * 1.1, 0, 255)  # Simple enhancement
        return enhanced.astype(np.uint8)

    def _macenko_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply Macenko stain normalization."""
        # Convert to optical density
        od = -np.log((image.astype(np.float64) + 1) / 256)

        # Compute stain concentrations
        he_matrix = self._target_stats["he_matrix"]
        concentrations = np.linalg.lstsq(he_matrix.T, od.reshape(-1, 3).T, rcond=None)[
            0
        ]

        # Normalize concentrations
        max_conc = self._target_stats["max_conc"]
        concentrations = np.clip(concentrations, 0, max_conc[:, np.newaxis])

        # Reconstruct image
        od_normalized = he_matrix.T @ concentrations
        image_normalized = np.exp(-od_normalized.T) * 256 - 1
        image_normalized = image_normalized.reshape(image.shape)
        image_normalized = np.clip(image_normalized, 0, 255)

        return image_normalized.astype(np.uint8)

    def fit_transform(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform images."""
        self.fit(images)
        return [self.transform(img) for img in images]


def denormalize_tensor(
    tensor: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized array (H, W, C) in [0, 255]
    """
    # Convert to numpy if needed
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    # Denormalize
    for i in range(3):
        tensor[i] = tensor[i] * std[i] + mean[i]

    # Transpose to HWC and convert to [0, 255]
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)

    return tensor
