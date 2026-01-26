"""Tensor operations for ComfyUI image handling."""
import torch


def to_comfy_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to ComfyUI image format (B, H, W, C) [0-1]."""
    if tensor.dim() == 3:  # (C, H, W)
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] <= 4:  # (B, C, H, W)
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor.clamp(0, 1)


def from_comfy_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI image (B, H, W, C) to (B, C, H, W)."""
    if tensor.dim() == 4 and tensor.shape[-1] <= 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 4D for batch operations."""
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    return tensor
