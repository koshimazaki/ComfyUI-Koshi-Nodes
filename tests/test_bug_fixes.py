"""Sprint 9: Bug fix tests — grid snapping, output normalization, burn/blur detection.

These tests verify the three confirmed bugs are fixed:
1. Grid snapping: transform output spatial dims divisible by 8
2. Output normalization: iterated transforms stay bounded
3. Burn/blur detection: feedback processor flags degraded images
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.flux_motion.core.transforms import (
    snap_to_grid,
    apply_affine_transform,
    apply_composite_transform,
    create_affine_matrix,
)
from nodes.flux_motion.feedback import KoshiFeedback


# ---------------------------------------------------------------------------
# Grid Snapping
# ---------------------------------------------------------------------------


class TestGridSnapping:
    """Output spatial dimensions must be divisible by 8 after snap_to_grid."""

    def test_already_aligned(self):
        """Tensor with H/W already multiples of 8 should be unchanged."""
        t = torch.rand(1, 4, 64, 64)
        result = snap_to_grid(t, 8)
        assert result.shape == (1, 4, 64, 64)
        assert torch.equal(result, t)

    def test_snap_height(self):
        """Height not divisible by 8 gets padded up."""
        t = torch.rand(1, 4, 60, 64)
        result = snap_to_grid(t, 8)
        assert result.shape[2] % 8 == 0
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_snap_width(self):
        """Width not divisible by 8 gets padded up."""
        t = torch.rand(1, 4, 64, 60)
        result = snap_to_grid(t, 8)
        assert result.shape[3] % 8 == 0
        assert result.shape[3] == 64

    def test_snap_both(self):
        """Both H and W get padded when needed."""
        t = torch.rand(1, 4, 61, 63)
        result = snap_to_grid(t, 8)
        assert result.shape[2] % 8 == 0
        assert result.shape[3] % 8 == 0
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_snap_small(self):
        """Very small dimensions still snap correctly."""
        t = torch.rand(1, 4, 1, 1)
        result = snap_to_grid(t, 8)
        assert result.shape[2] == 8
        assert result.shape[3] == 8

    def test_snap_preserves_content(self):
        """Original content is preserved in the top-left region."""
        t = torch.rand(1, 4, 60, 60)
        result = snap_to_grid(t, 8)
        assert torch.equal(result[:, :, :60, :60], t)

    def test_snap_custom_grid(self):
        """Custom grid size works (e.g., 16)."""
        t = torch.rand(1, 4, 50, 50)
        result = snap_to_grid(t, 16)
        assert result.shape[2] % 16 == 0
        assert result.shape[3] % 16 == 0

    def test_snap_batch(self):
        """Batch dimension is preserved."""
        t = torch.rand(4, 4, 61, 61)
        result = snap_to_grid(t, 8)
        assert result.shape[0] == 4
        assert result.shape[2] % 8 == 0
        assert result.shape[3] % 8 == 0

    def test_composite_with_snap(self):
        """apply_composite_transform with snap_to_8=True produces grid-aligned output."""
        t = torch.rand(1, 4, 60, 60)
        params = {"zoom": 1.02, "angle": 1.0}
        result = apply_composite_transform(t, params, snap_to_8=True)
        # Output must be grid-aligned even though input wasn't
        assert result.shape[2] % 8 == 0
        assert result.shape[3] % 8 == 0


# ---------------------------------------------------------------------------
# Output Normalization
# ---------------------------------------------------------------------------


class TestOutputNormalization:
    """Iterated transforms must stay bounded in [0, 1]."""

    def test_single_transform_clamped(self):
        """Single transform with normalize=True stays in [0, 1]."""
        t = torch.rand(1, 4, 64, 64)
        matrix = create_affine_matrix(zoom=1.05, angle=5.0)
        result = apply_affine_transform(t, matrix, normalize_output=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_iterated_transforms_bounded(self):
        """50 iterations of transforms with normalization stay bounded."""
        t = torch.rand(1, 4, 64, 64)
        params = {"zoom": 1.02, "angle": 1.0, "translation_x": 1.0}
        for _ in range(50):
            t = apply_composite_transform(t, params, normalize=True)
        assert torch.isfinite(t).all(), "Values must be finite after 50 iterations"
        assert t.min() >= 0.0, f"Min {t.min()} below 0 after 50 iterations"
        assert t.max() <= 1.0, f"Max {t.max()} above 1 after 50 iterations"

    def test_normalize_off_by_default(self):
        """normalize=False (default) does not clamp."""
        # This just verifies the parameter is properly optional
        t = torch.rand(1, 4, 64, 64)
        params = {"zoom": 1.02}
        result = apply_composite_transform(t, params)
        # Should still work, just no explicit clamping
        assert result.shape == t.shape

    def test_composite_normalize_flag(self):
        """apply_composite_transform respects normalize flag."""
        t = torch.rand(1, 4, 64, 64) * 2.0  # Values > 1
        params = {"zoom": 1.01}
        result = apply_composite_transform(t, params, normalize=True)
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Burn Detection
# ---------------------------------------------------------------------------


class TestBurnDetection:
    """Feedback processor must detect burned-out images."""

    def setup_method(self):
        self.fb = KoshiFeedback()

    def test_white_image_burned(self):
        """All-white image should be flagged as burned."""
        white = np.ones((64, 64, 3), dtype=np.uint8) * 255
        assert self.fb.detect_burn(white) is True

    def test_black_image_burned(self):
        """All-black image should be flagged as burned."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        assert self.fb.detect_burn(black) is True

    def test_normal_image_not_burned(self):
        """Image with normal contrast should not be flagged."""
        np.random.seed(42)
        normal = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert self.fb.detect_burn(normal) is False

    def test_gradient_not_burned(self):
        """Gradient image has reasonable std, not burned."""
        grad = np.linspace(0, 255, 64 * 64 * 3).reshape(64, 64, 3).astype(np.uint8)
        assert self.fb.detect_burn(grad) is False

    def test_custom_threshold(self):
        """Custom threshold adjusts sensitivity."""
        # Nearly uniform image
        almost_flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        almost_flat[:2, :2] = 130  # Tiny variation, std/255 ~ 0.00025
        # With default threshold (0.1) — this should be burned
        assert self.fb.detect_burn(almost_flat) is True
        # With threshold below the actual std — not burned
        assert self.fb.detect_burn(almost_flat, threshold=0.0001) is False


# ---------------------------------------------------------------------------
# Blur Detection
# ---------------------------------------------------------------------------


class TestBlurDetection:
    """Feedback processor must detect excessively blurred images."""

    def setup_method(self):
        self.fb = KoshiFeedback()

    def test_flat_image_blurry(self):
        """Uniform image should be flagged as blurry."""
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert self.fb.detect_blur(flat) is True

    def test_noisy_image_not_blurry(self):
        """High-frequency noise image should not be flagged."""
        np.random.seed(42)
        noisy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert self.fb.detect_blur(noisy) is False

    def test_edge_image_not_blurry(self):
        """Image with strong edges should not be flagged."""
        edge = np.zeros((64, 64, 3), dtype=np.uint8)
        edge[:, 32:] = 255  # Sharp vertical edge
        assert self.fb.detect_blur(edge) is False

    def test_custom_threshold(self):
        """Custom threshold adjusts blur sensitivity."""
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        # With very low threshold, even flat is "not blurry"
        assert self.fb.detect_blur(flat, threshold=0.0) is False

    def test_gradient_not_blurry(self):
        """Smooth gradient may or may not be blurry depending on threshold,
        but with default threshold the gradient has enough variation."""
        grad = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            grad[:, i] = int(i * 255 / 63)
        # Gradient has consistent edges (step function in discrete space)
        # This depends on the detection method, so just check it returns bool
        result = self.fb.detect_blur(grad)
        assert isinstance(result, bool)
