"""Grad-CAM implementation for explainability validation."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN explainability."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: Union[str, List[str]],
        use_cuda: bool = True,
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layers: Names of target layers for CAM generation
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.gradients = {}
        self.activations = {}
        self.handles = []
        
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        
        def get_gradients_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook

        def get_activations_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks on target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handle_forward = module.register_forward_hook(
                    get_activations_hook(name)
                )
                handle_backward = module.register_full_backward_hook(
                    get_gradients_hook(name)
                )
                self.handles.extend([handle_forward, handle_backward])

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor (batch_size, channels, height, width)
            class_idx: Target class index (None for predicted class)
            retain_graph: Whether to retain computation graph
            normalize: Whether to normalize the heatmap

        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # Forward pass
        input_tensor.requires_grad_()
        output = self.model(input_tensor)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[:, class_idx].sum()
        class_score.backward(retain_graph=retain_graph)
        
        # Generate CAM for each target layer
        cams = []
        for layer_name in self.target_layers:
            if layer_name in self.gradients and layer_name in self.activations:
                cam = self._compute_cam(
                    self.gradients[layer_name],
                    self.activations[layer_name],
                    normalize=normalize
                )
                cams.append(cam)
        
        if len(cams) == 1:
            return cams[0]
        elif len(cams) > 1:
            # Average multiple CAMs
            return np.mean(cams, axis=0)
        else:
            raise ValueError("No CAMs generated. Check target layer names.")

    def _compute_cam(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute CAM from gradients and activations.

        Args:
            gradients: Gradient tensor
            activations: Activation tensor
            normalize: Whether to normalize the CAM

        Returns:
            CAM as numpy array
        """
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=(224, 224),  # Assuming input size
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if normalize:
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam

    def __del__(self):
        """Clean up hooks."""
        for handle in self.handles:
            handle.remove()


class GradCAMPlusPlus(GradCAM):
    """Improved Grad-CAM with better localization for multiple objects."""

    def _compute_cam(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ from gradients and activations.

        Args:
            gradients: Gradient tensor
            activations: Activation tensor
            normalize: Whether to normalize the CAM

        Returns:
            CAM as numpy array
        """
        # Compute alpha weights (Grad-CAM++ improvement)
        grad_2 = gradients.pow(2)
        grad_3 = grad_2 * gradients
        
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(dim=(2, 3), keepdim=True))
        
        # Weighted combination
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU and resize
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy and normalize
        cam = cam.squeeze().cpu().numpy()
        
        if normalize:
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam


def inject_discriminative_patch(
    image: torch.Tensor,
    roi: Tuple[int, int, int, int],
    intensity: float = 1.0,
    pattern: str = "solid"
) -> torch.Tensor:
    """
    Inject discriminative patch for testing explainability.

    Args:
        image: Input image tensor (batch_size, channels, height, width)
        roi: Region of interest as (y1, y2, x1, x2)
        intensity: Patch intensity multiplier
        pattern: Patch pattern ("solid", "checkerboard", "gradient")

    Returns:
        Image with injected patch
    """
    patched = image.clone()
    y1, y2, x1, x2 = roi
    
    if pattern == "solid":
        patched[:, :, y1:y2, x1:x2] = intensity
    elif pattern == "checkerboard":
        # Create checkerboard pattern
        h, w = y2 - y1, x2 - x1
        checkerboard = torch.zeros((h, w))
        checkerboard[::2, ::2] = 1
        checkerboard[1::2, 1::2] = 1
        patched[:, :, y1:y2, x1:x2] = intensity * checkerboard.unsqueeze(0)
    elif pattern == "gradient":
        # Create gradient pattern
        h, w = y2 - y1, x2 - x1
        gradient = torch.linspace(0, intensity, w).unsqueeze(0).expand(h, -1)
        patched[:, :, y1:y2, x1:x2] = gradient.unsqueeze(0)
    
    return patched


def compute_localization_metrics(
    cam: np.ndarray,
    roi: Tuple[int, int, int, int],
    threshold: float = 0.5
) -> dict:
    """
    Compute localization metrics for CAM evaluation.

    Args:
        cam: CAM heatmap
        roi: Ground truth region of interest
        threshold: Threshold for binary mask

    Returns:
        Dictionary of metrics
    """
    y1, y2, x1, x2 = roi
    
    # Create binary mask from CAM
    cam_binary = (cam > threshold).astype(np.uint8)
    
    # Create ground truth mask
    gt_mask = np.zeros_like(cam, dtype=np.uint8)
    gt_mask[y1:y2, x1:x2] = 1
    
    # Compute intersection and union
    intersection = np.logical_and(cam_binary, gt_mask).sum()
    union = np.logical_or(cam_binary, gt_mask).sum()
    
    # Compute metrics
    iou = intersection / union if union > 0 else 0
    precision = intersection / cam_binary.sum() if cam_binary.sum() > 0 else 0
    recall = intersection / gt_mask.sum() if gt_mask.sum() > 0 else 0
    
    # Pointing game metric (max activation in ROI)
    roi_max = cam[y1:y2, x1:x2].max()
    global_max = cam.max()
    pointing_acc = 1.0 if roi_max == global_max else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'pointing_accuracy': pointing_acc,
        'roi_mean_activation': cam[y1:y2, x1:x2].mean(),
        'background_mean_activation': cam[gt_mask == 0].mean() if (gt_mask == 0).sum() > 0 else 0,
    }