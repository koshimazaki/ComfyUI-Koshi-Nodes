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
def snap_to_grid(tensor: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
    """
    Pad tensor spatial dimensions to the nearest multiple of grid_size.

    Args:
        tensor: Input tensor (B, C, H, W)
        grid_size: Grid alignment size (default 8 for VAE compatibility)

    Returns:
        Tensor with H/W padded to multiples of grid_size
    """
    _, _, h, w = tensor.shape
    new_h = ((h + grid_size - 1) // grid_size) * grid_size
    new_w = ((w + grid_size - 1) // grid_size) * grid_size
    if new_h == h and new_w == w:
        return tensor
    pad_w = new_w - w
    pad_h = new_h - h
    # Reflect padding requires pad < dim; fall back to replicate for small tensors
    if pad_w >= w or pad_h >= h:
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')


@torch.no_grad()
def apply_affine_transform(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'reflection',
    normalize_output: bool = False
) -> torch.Tensor:
    """
    Apply affine transformation to a tensor.

    Args:
        tensor: Input tensor (B, C, H, W)
        matrix: Affine matrix (2, 3) or (B, 2, 3)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')
        normalize_output: Clamp output to [0, 1] to prevent drift

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

    if normalize_output:
        output = torch.clamp(output, 0.0, 1.0)

    return output


@torch.no_grad()
def apply_composite_transform(
    tensor: torch.Tensor,
    motion_params: Dict[str, float],
    snap_to_8: bool = False,
    normalize: bool = False
) -> torch.Tensor:
    """
    Apply multiple transforms in sequence (Deforum order: translate -> rotate -> zoom).

    Args:
        tensor: Input tensor (B, C, H, W)
        motion_params: Dictionary with zoom, angle, translation_x, translation_y
        snap_to_8: Snap output dimensions to multiples of 8
        normalize: Clamp output to [0, 1] to prevent drift over iterations

    Returns:
        Transformed tensor
    """
    batch_size, channels, height, width = tensor.shape

    zoom = motion_params.get("zoom", 1.0)
    angle = motion_params.get("angle", 0.0)
    tx = motion_params.get("translation_x", 0.0)
    ty = motion_params.get("translation_y", 0.0)

    if zoom == 1.0 and angle == 0.0 and tx == 0.0 and ty == 0.0:
        result = tensor
    else:
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
        result = apply_affine_transform(tensor, matrix, normalize_output=normalize)

    if snap_to_8:
        result = snap_to_grid(result, 8)

    return result


__all__ = [
    "create_affine_matrix",
    "apply_affine_transform",
    "apply_composite_transform",
    "snap_to_grid",
]
