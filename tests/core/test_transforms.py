"""Tests for nodes.flux_motion.core.transforms module.

Covers:
  - create_affine_matrix: identity, zoom, rotation, translation, device/dtype
  - apply_affine_transform: shape preservation, identity invariance, padding modes
  - apply_composite_transform: dict-based API, identity shortcut, combined params
"""

import sys
import os
import math
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nodes.flux_motion.core.transforms import (
    create_affine_matrix,
    apply_affine_transform,
    apply_composite_transform,
)


# -----------------------------------------------------------------------
# create_affine_matrix
# -----------------------------------------------------------------------

class TestCreateAffineMatrix:
    """Tests for affine matrix construction."""

    def test_identity_matrix(self):
        """Default params (zoom=1, angle=0, tx=0, ty=0) produce identity-like 2x3."""
        mat = create_affine_matrix(zoom=1.0, angle=0.0, translation_x=0.0, translation_y=0.0)
        assert mat.shape == (2, 3)
        expected = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], dtype=torch.float32)
        assert torch.allclose(mat, expected, atol=1e-6)

    def test_zoom_scales_diagonal(self):
        """Zoom=2 should double the top-left 2x2 diagonal (no rotation)."""
        mat = create_affine_matrix(zoom=2.0, angle=0.0)
        assert mat[0, 0].item() == pytest.approx(2.0, abs=1e-6)
        assert mat[1, 1].item() == pytest.approx(2.0, abs=1e-6)
        # Off-diagonals should be zero (no rotation)
        assert mat[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert mat[1, 0].item() == pytest.approx(0.0, abs=1e-6)

    def test_rotation_90_degrees(self):
        """90-degree rotation should produce cos(90)=0, sin(90)=1."""
        mat = create_affine_matrix(zoom=1.0, angle=90.0)
        cos_90 = math.cos(math.radians(90.0))  # ~0
        sin_90 = math.sin(math.radians(90.0))  # ~1
        assert mat[0, 0].item() == pytest.approx(cos_90, abs=1e-5)
        assert mat[0, 1].item() == pytest.approx(-sin_90, abs=1e-5)
        assert mat[1, 0].item() == pytest.approx(sin_90, abs=1e-5)
        assert mat[1, 1].item() == pytest.approx(cos_90, abs=1e-5)

    def test_rotation_45_degrees(self):
        """45-degree rotation with zoom=1 should have cos=sin=~0.7071."""
        mat = create_affine_matrix(zoom=1.0, angle=45.0)
        val = math.cos(math.radians(45.0))
        assert mat[0, 0].item() == pytest.approx(val, abs=1e-5)
        assert mat[1, 1].item() == pytest.approx(val, abs=1e-5)

    def test_translation_normalised_by_dimensions(self):
        """translation_x / width * 2 should appear in mat[0, 2]."""
        mat = create_affine_matrix(
            translation_x=32.0, translation_y=16.0, width=64, height=64
        )
        assert mat[0, 2].item() == pytest.approx(32.0 / 64 * 2, abs=1e-6)
        assert mat[1, 2].item() == pytest.approx(16.0 / 64 * 2, abs=1e-6)

    def test_zero_translation_in_identity(self):
        mat = create_affine_matrix()
        assert mat[0, 2].item() == pytest.approx(0.0, abs=1e-6)
        assert mat[1, 2].item() == pytest.approx(0.0, abs=1e-6)

    def test_output_shape(self):
        mat = create_affine_matrix()
        assert mat.shape == (2, 3)

    def test_output_dtype_default(self):
        mat = create_affine_matrix()
        assert mat.dtype == torch.float32

    def test_custom_dtype(self):
        mat = create_affine_matrix(dtype=torch.float64)
        assert mat.dtype == torch.float64

    def test_negative_angle(self):
        """Negative angle should rotate clockwise."""
        mat_pos = create_affine_matrix(angle=30.0)
        mat_neg = create_affine_matrix(angle=-30.0)
        # sin component should flip sign
        assert mat_pos[1, 0].item() == pytest.approx(-mat_neg[1, 0].item(), abs=1e-5)


# -----------------------------------------------------------------------
# apply_affine_transform
# -----------------------------------------------------------------------

class TestApplyAffineTransform:
    """Tests for applying affine matrix to tensors."""

    def test_output_shape_matches_input(self):
        tensor = torch.randn(1, 3, 32, 32)
        mat = create_affine_matrix()
        result = apply_affine_transform(tensor, mat)
        assert result.shape == tensor.shape

    def test_identity_preserves_content(self):
        """Identity matrix should return approximately the same tensor."""
        torch.manual_seed(42)
        tensor = torch.randn(1, 3, 16, 16)
        mat = create_affine_matrix()
        result = apply_affine_transform(tensor, mat)
        assert torch.allclose(tensor, result, atol=1e-4)

    def test_batch_dimension_preserved(self):
        tensor = torch.randn(4, 3, 16, 16)
        mat = create_affine_matrix()
        result = apply_affine_transform(tensor, mat)
        assert result.shape[0] == 4

    def test_batched_matrix_input(self):
        """Matrix with batch dim (B, 2, 3) should work."""
        tensor = torch.randn(2, 3, 16, 16)
        mat = create_affine_matrix().unsqueeze(0).expand(2, -1, -1)
        result = apply_affine_transform(tensor, mat)
        assert result.shape == tensor.shape

    def test_padding_mode_zeros(self):
        """Zeros padding should not crash and produce valid output."""
        tensor = torch.ones(1, 1, 8, 8)
        mat = create_affine_matrix(translation_x=100.0, width=8)
        result = apply_affine_transform(tensor, mat, padding_mode='zeros')
        assert result.shape == tensor.shape

    def test_padding_mode_border(self):
        tensor = torch.ones(1, 1, 8, 8)
        mat = create_affine_matrix(translation_x=4.0, width=8)
        result = apply_affine_transform(tensor, mat, padding_mode='border')
        assert result.shape == tensor.shape

    def test_padding_mode_reflection(self):
        tensor = torch.ones(1, 1, 8, 8)
        mat = create_affine_matrix(translation_x=4.0, width=8)
        result = apply_affine_transform(tensor, mat, padding_mode='reflection')
        assert result.shape == tensor.shape

    def test_interpolation_mode_nearest(self):
        tensor = torch.randn(1, 1, 8, 8)
        mat = create_affine_matrix(zoom=1.5)
        result = apply_affine_transform(tensor, mat, mode='nearest')
        assert result.shape == tensor.shape

    def test_no_grad_context(self):
        """Transform should not track gradients."""
        tensor = torch.randn(1, 1, 8, 8, requires_grad=False)
        mat = create_affine_matrix()
        result = apply_affine_transform(tensor, mat)
        assert not result.requires_grad


# -----------------------------------------------------------------------
# apply_composite_transform
# -----------------------------------------------------------------------

class TestApplyCompositeTransform:
    """Tests for the high-level dict-based transform API."""

    def test_identity_params_return_same_tensor(self):
        """All-default params should return the exact same tensor object (early exit)."""
        tensor = torch.randn(1, 3, 16, 16)
        result = apply_composite_transform(tensor, {
            "zoom": 1.0, "angle": 0.0,
            "translation_x": 0.0, "translation_y": 0.0,
        })
        # The implementation returns tensor directly for identity
        assert result is tensor

    def test_empty_dict_returns_same_tensor(self):
        """Empty dict defaults to identity -- should early-exit."""
        tensor = torch.randn(1, 3, 16, 16)
        result = apply_composite_transform(tensor, {})
        assert result is tensor

    def test_zoom_changes_output(self):
        torch.manual_seed(42)
        tensor = torch.randn(1, 1, 16, 16)
        result = apply_composite_transform(tensor, {"zoom": 2.0})
        # Output should differ from input
        assert not torch.allclose(tensor, result, atol=1e-4)

    def test_rotation_changes_output(self):
        torch.manual_seed(42)
        tensor = torch.randn(1, 1, 16, 16)
        result = apply_composite_transform(tensor, {"angle": 45.0})
        assert not torch.allclose(tensor, result, atol=1e-4)

    def test_output_shape_preserved(self):
        tensor = torch.randn(2, 4, 32, 32)
        result = apply_composite_transform(tensor, {"zoom": 1.5, "angle": 10.0})
        assert result.shape == tensor.shape

    def test_combined_params(self):
        """Zoom + rotation + translation should produce valid output."""
        tensor = torch.randn(1, 3, 16, 16)
        result = apply_composite_transform(tensor, {
            "zoom": 1.2,
            "angle": 15.0,
            "translation_x": 2.0,
            "translation_y": -1.0,
        })
        assert result.shape == tensor.shape
        assert torch.isfinite(result).all()

    def test_partial_params_defaults(self):
        """Only specifying zoom should leave angle/translation at defaults."""
        tensor = torch.randn(1, 1, 8, 8)
        result = apply_composite_transform(tensor, {"zoom": 0.5})
        assert result.shape == tensor.shape
        assert torch.isfinite(result).all()

    def test_output_is_finite(self):
        """Even with extreme params, output should not contain NaN or Inf."""
        tensor = torch.randn(1, 1, 8, 8)
        result = apply_composite_transform(tensor, {
            "zoom": 0.01, "angle": 359.0,
            "translation_x": 100.0, "translation_y": -100.0,
        })
        assert torch.isfinite(result).all()
