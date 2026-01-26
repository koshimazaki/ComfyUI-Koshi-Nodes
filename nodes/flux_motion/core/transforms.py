"""Geometric transforms for latent space motion."""

from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def create_affine_matrix(
    zoom: float = 1.0,
    angle: float = 0.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    width: int = 64,
    height: int = 64,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create 2D affine transformation matrix.

    Args:
        zoom: Scale factor (1.0 = no zoom, >1 = zoom in)
        angle: Rotation in degrees (positive = counter-clockwise)
        translation_x: Horizontal translation in pixels
        translation_y: Vertical translation in pixels
        width: Image width for normalizing translation
        height: Image height for normalizing translation

    Returns:
        Affine matrix of shape (2, 3)
    """
    angle_rad = angle * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    tx_norm = translation_x / width * 2
    ty_norm = translation_y / height * 2

    matrix = torch.tensor([
        [zoom * cos_a, -zoom * sin_a, tx_norm],
        [zoom * sin_a,  zoom * cos_a, ty_norm]
    ], device=device, dtype=dtype)

    return matrix


@torch.no_grad()
def apply_affine_transform(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'reflection'
) -> torch.Tensor:
    """
    Apply affine transformation to a tensor.

    Args:
        tensor: Input tensor (B, C, H, W)
        matrix: Affine matrix (2, 3) or (B, 2, 3)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')

    Returns:
        Transformed tensor
    """
    batch_size = tensor.shape[0]

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)

    grid = F.affine_grid(matrix, tensor.size(), align_corners=False)
    output = F.grid_sample(
        tensor, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    )

    return output


@torch.no_grad()
def apply_composite_transform(
    tensor: torch.Tensor,
    motion_params: Dict[str, float]
) -> torch.Tensor:
    """
    Apply multiple transforms in sequence (Deforum order: translate -> rotate -> zoom).

    Args:
        tensor: Input tensor (B, C, H, W)
        motion_params: Dictionary with zoom, angle, translation_x, translation_y

    Returns:
        Transformed tensor
    """
    batch_size, channels, height, width = tensor.shape

    zoom = motion_params.get("zoom", 1.0)
    angle = motion_params.get("angle", 0.0)
    tx = motion_params.get("translation_x", 0.0)
    ty = motion_params.get("translation_y", 0.0)

    if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
        return tensor

    matrix = create_affine_matrix(
        zoom=zoom,
        angle=angle,
        translation_x=tx,
        translation_y=ty,
        width=width,
        height=height,
        device=tensor.device,
        dtype=tensor.dtype
    )

    return apply_affine_transform(tensor, matrix)


__all__ = [
    "create_affine_matrix",
    "apply_affine_transform",
    "apply_composite_transform",
]
