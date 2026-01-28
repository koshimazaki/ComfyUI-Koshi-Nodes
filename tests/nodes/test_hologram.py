"""Tests for nodes.effects.hologram -- KoshiHologram, KoshiScanlines, KoshiVideoGlitch."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch

from nodes.effects.hologram import KoshiHologram, KoshiScanlines, KoshiVideoGlitch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_image():
    """Single 64x64 RGB image [1, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def batch_image():
    """Batch of 4 RGB images [4, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Default hologram parameters
# ---------------------------------------------------------------------------

HOLOGRAM_DEFAULTS = dict(
    color_preset="cyan",
    scanline_intensity=0.3,
    scanline_count=100,
    glitch_intensity=0.1,
    edge_glow=0.5,
    grid_opacity=0.2,
    grid_size=20,
    alpha=0.9,
    time=0.0,
)


# ===================================================================
# KoshiHologram tests
# ===================================================================

class TestKoshiHologramShape:
    """Output must be [B, H, W, 3] matching the input spatial dimensions."""

    def test_output_shape_matches_input(self, single_image):
        node = KoshiHologram()
        result = node.apply(single_image, **HOLOGRAM_DEFAULTS)
        output = result[0]
        assert output.shape == single_image.shape


class TestKoshiHologramRange:
    """Output pixel values must stay within [0, 1]."""

    def test_output_within_unit_range(self, single_image):
        node = KoshiHologram()
        result = node.apply(single_image, **HOLOGRAM_DEFAULTS)
        output = result[0]
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6


class TestKoshiHologramColorPresets:
    """Every named color preset must produce valid output without error."""

    @pytest.mark.parametrize("preset", ["cyan", "red_error", "green_matrix", "purple", "orange", "white"])
    def test_color_preset_runs(self, preset, single_image):
        node = KoshiHologram()
        params = dict(HOLOGRAM_DEFAULTS)
        params["color_preset"] = preset
        result = node.apply(single_image, **params)
        output = result[0]
        assert output.shape == single_image.shape
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6


class TestKoshiHologramBatch:
    """All frames in a batch must be processed."""

    def test_batch_shape(self, batch_image):
        node = KoshiHologram()
        result = node.apply(batch_image, **HOLOGRAM_DEFAULTS)
        output = result[0]
        assert output.shape == batch_image.shape


# ===================================================================
# KoshiScanlines tests
# ===================================================================

class TestScanlinesHorizontal:
    """Horizontal scanlines must preserve tensor shape."""

    def test_horizontal_output_shape(self, single_image):
        node = KoshiScanlines()
        result = node.apply(single_image, count=100, intensity=0.3, direction="horizontal", animate_speed=0.0)
        output = result[0]
        assert output.shape == single_image.shape


class TestScanlinesVertical:
    """Vertical scanlines must preserve tensor shape."""

    def test_vertical_output_shape(self, single_image):
        node = KoshiScanlines()
        result = node.apply(single_image, count=100, intensity=0.3, direction="vertical", animate_speed=0.0)
        output = result[0]
        assert output.shape == single_image.shape


class TestScanlinesZeroIntensity:
    """Zero intensity scanlines should leave the image nearly unchanged."""

    def test_zero_intensity_identity(self, single_image):
        node = KoshiScanlines()
        result = node.apply(single_image, count=100, intensity=0.0, direction="horizontal", animate_speed=0.0)
        output = result[0]
        assert torch.allclose(output, single_image, atol=1e-5)


class TestScanlinesOutputRange:
    """Scanline output must be in [0, 1]."""

    def test_output_range(self, single_image):
        node = KoshiScanlines()
        result = node.apply(single_image, count=200, intensity=1.0, direction="horizontal", animate_speed=0.0)
        output = result[0]
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6


# ===================================================================
# KoshiVideoGlitch tests
# ===================================================================

class TestVideoGlitchShape:
    """Output tensor must match input dimensions."""

    def test_output_shape(self, single_image):
        node = KoshiVideoGlitch()
        result = node.apply(single_image, distortion=0.1, distortion2=0.04, speed=1.0, seed=42)
        output = result[0]
        assert output.shape == single_image.shape


class TestVideoGlitchSeedDeterminism:
    """Same seed and parameters must produce identical output."""

    def test_same_seed_same_output(self, single_image):
        node = KoshiVideoGlitch()
        result_a = node.apply(single_image, distortion=0.1, distortion2=0.04, speed=1.0, seed=42)
        result_b = node.apply(single_image, distortion=0.1, distortion2=0.04, speed=1.0, seed=42)
        assert torch.allclose(result_a[0], result_b[0], atol=1e-6)


class TestVideoGlitchDifferentSeeds:
    """Different seeds must produce different output."""

    def test_different_seeds_differ(self, single_image):
        node = KoshiVideoGlitch()
        result_a = node.apply(single_image, distortion=0.1, distortion2=0.04, speed=1.0, seed=42)
        result_b = node.apply(single_image, distortion=0.1, distortion2=0.04, speed=1.0, seed=999)
        assert not torch.allclose(result_a[0], result_b[0], atol=1e-6)


class TestVideoGlitchOutputRange:
    """Output values must be in [0, 1]."""

    def test_output_range(self, single_image):
        node = KoshiVideoGlitch()
        result = node.apply(single_image, distortion=0.3, distortion2=0.1, speed=2.0, seed=42)
        output = result[0]
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6
