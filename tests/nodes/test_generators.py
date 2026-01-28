"""Tests for generator nodes: KoshiGlitchCandies, KoshiShapeMorph,
KoshiNoiseDisplace."""

import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch
import numpy as np

from nodes.generators.glitch_candies import (
    KoshiGlitchCandies,
    KoshiShapeMorph,
    KoshiNoiseDisplace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(batch=1, h=64, w=64, c=3):
    """Create a ComfyUI-format image tensor [B, H, W, C] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(batch, h, w, c, dtype=torch.float32)


# 2D patterns that do not use raymarching
PATTERNS_2D = [
    "waves", "circles", "plasma", "voronoi", "checkerboard", "swirl", "ripple",
]

# Raymarched 3D patterns
PATTERNS_RM = ["rm_cube", "rm_sphere", "rm_torus"]


# ===================================================================
# 1. KoshiGlitchCandies - Shape & value tests
# ===================================================================

class TestGlitchCandiesPatterns2D:
    """2D patterns produce correct output shape and range."""

    @pytest.mark.parametrize("pattern", PATTERNS_2D)
    def test_2d_pattern_output_shape(self, pattern):
        """Each 2D pattern must produce [B, H, W, 3] image and [B, H, W] mask."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=64, height=64, pattern=pattern,
            time=0.5, scale=1.0, seed=0,
        )
        image = result[0]
        mask = result[1]
        assert image.shape == (1, 64, 64, 3)
        assert mask.shape == (1, 64, 64)


class TestGlitchCandiesPatternsRM:
    """Raymarched 3D patterns produce correct output shape."""

    @pytest.mark.parametrize("pattern", PATTERNS_RM)
    def test_rm_pattern_output_shape(self, pattern):
        """Raymarched patterns must produce [B, H, W, 3] image and [B, H, W] mask."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=64, height=64, pattern=pattern,
            time=0.0, scale=1.0, seed=0,
            camera_distance=3.0, rotation_x=0.0, rotation_y=0.0,
        )
        image = result[0]
        mask = result[1]
        assert image.shape == (1, 64, 64, 3)
        assert mask.shape == (1, 64, 64)


class TestGlitchCandiesOutputFormat:
    """Validate tensor properties of generated output."""

    def test_image_shape_bhwc(self):
        """Image output must be [B, H, W, 3]."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=128, height=96, pattern="waves",
            time=0.0, scale=1.0, seed=0,
        )
        assert result[0].shape == (1, 96, 128, 3)

    def test_mask_shape_bhw(self):
        """Mask output must be [B, H, W]."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=128, height=96, pattern="waves",
            time=0.0, scale=1.0, seed=0,
        )
        assert result[1].shape == (1, 96, 128)

    def test_values_in_unit_range(self):
        """All output values must be in [0, 1]."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=64, height=64, pattern="plasma",
            time=1.0, scale=2.0, seed=7,
        )
        image = result[0]
        mask = result[1]
        assert image.min().item() >= 0.0
        assert image.max().item() <= 1.0
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0


class TestGlitchCandiesBatch:
    """Batch and loop frame generation."""

    def test_batch_size_greater_than_one(self):
        """batch_size=3 should produce 3 frames."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=64, height=64, pattern="circles",
            time=0.0, scale=1.0, seed=0,
            batch_size=3,
        )
        assert result[0].shape[0] == 3
        assert result[1].shape[0] == 3

    def test_loop_frames(self):
        """loop_frames=4 should produce exactly 4 frames."""
        node = KoshiGlitchCandies()
        result = node.generate(
            width=64, height=64, pattern="checkerboard",
            time=0.0, scale=1.0, seed=0,
            loop_frames=4,
        )
        assert result[0].shape[0] == 4
        assert result[1].shape[0] == 4

    def test_seed_determinism(self):
        """Same seed must produce identical output."""
        node = KoshiGlitchCandies()
        r1 = node.generate(
            width=64, height=64, pattern="swirl",
            time=1.0, scale=1.0, seed=123,
        )
        r2 = node.generate(
            width=64, height=64, pattern="swirl",
            time=1.0, scale=1.0, seed=123,
        )
        assert torch.allclose(r1[0], r2[0])
        assert torch.allclose(r1[1], r2[1])


# ===================================================================
# 2. KoshiShapeMorph
# ===================================================================

class TestKoshiShapeMorph:
    """Morphing between two images."""

    def test_blend_zero_equals_image_a(self):
        """blend=0.0 should return image_a (for linear mode)."""
        node = KoshiShapeMorph()
        a = _make_image(batch=1, h=32, w=32)
        b = torch.ones(1, 32, 32, 3, dtype=torch.float32)
        result = node.morph(image_a=a, image_b=b, blend=0.0, blend_mode="linear")
        output = result[0]
        assert torch.allclose(output, a, atol=1e-6)

    def test_blend_one_equals_image_b(self):
        """blend=1.0 should return image_b (for linear mode)."""
        node = KoshiShapeMorph()
        a = torch.zeros(1, 32, 32, 3, dtype=torch.float32)
        b = _make_image(batch=1, h=32, w=32)
        result = node.morph(image_a=a, image_b=b, blend=1.0, blend_mode="linear")
        output = result[0]
        assert torch.allclose(output, b, atol=1e-6)

    @pytest.mark.parametrize("mode", ["linear", "smooth", "ease_in", "ease_out", "sine"])
    def test_all_blend_modes_run(self, mode):
        """Every blend mode should execute without error."""
        node = KoshiShapeMorph()
        a = _make_image(batch=1, h=32, w=32)
        b = _make_image(batch=1, h=32, w=32)
        result = node.morph(image_a=a, image_b=b, blend=0.5, blend_mode=mode)
        assert result[0].shape == (1, 32, 32, 3)
        assert result[1].shape == (1, 32, 32)


# ===================================================================
# 3. KoshiNoiseDisplace
# ===================================================================

class TestKoshiNoiseDisplace:
    """Noise displacement output shape."""

    def test_output_shape_matches_input(self):
        """Displaced image should have same shape as input."""
        node = KoshiNoiseDisplace()
        image = _make_image(batch=2, h=64, w=64)
        result = node.displace(
            image=image,
            strength=0.1,
            scale=10.0,
            octaves=4,
            seed=0,
        )
        output = result[0]
        assert output.shape == image.shape
