"""Tests for nodes.effects.chromatic_aberration -- KoshiChromaticAberration CPU fallback and contracts."""

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

from nodes.effects.chromatic_aberration import KoshiChromaticAberration, SCIPY_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    """KoshiChromaticAberration forced to CPU path."""
    n = KoshiChromaticAberration()
    n.use_gpu = False
    return n


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


# ===================================================================
# 1. Output shape matches input
# ===================================================================

class TestOutputShape:
    """Chromatic aberration must preserve tensor dimensions."""

    def test_shape_preserved(self, node, single_image):
        result = node.apply(single_image, intensity=1.0, red_offset=1.0, green_offset=0.0, blue_offset=-1.0)
        output = result[0]
        assert output.shape == single_image.shape


# ===================================================================
# 2. Output in [0, 1] range
# ===================================================================

class TestOutputRange:
    """Pixel values must remain within valid range after processing."""

    def test_output_clamped(self, node, single_image):
        result = node.apply(single_image, intensity=5.0, red_offset=5.0, green_offset=-3.0, blue_offset=-5.0)
        output = result[0]
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ===================================================================
# 3. Zero intensity: output approximately equals input
# ===================================================================

class TestZeroIntensity:
    """All offsets at zero with zero intensity means no channel separation."""

    def test_zero_offsets_identity(self, node, single_image):
        result = node.apply(single_image, intensity=0.0, red_offset=0.0, green_offset=0.0, blue_offset=0.0)
        output = result[0]
        assert torch.allclose(output, single_image, atol=1e-4)


# ===================================================================
# 4. Non-zero offsets change individual channels
# ===================================================================

class TestChannelSeparation:
    """With significant offsets, channels should shift differently from original."""

    def test_channels_differ_from_input(self, node, single_image):
        result = node.apply(single_image, intensity=5.0, red_offset=5.0, green_offset=0.0, blue_offset=-5.0)
        output = result[0]
        # Red and blue channels should differ from the original
        red_diff = (output[..., 0] - single_image[..., 0]).abs().mean().item()
        blue_diff = (output[..., 2] - single_image[..., 2]).abs().mean().item()
        assert red_diff > 1e-4, "Red channel should shift with non-zero red_offset"
        assert blue_diff > 1e-4, "Blue channel should shift with non-zero blue_offset"


# ===================================================================
# 5. Batch processing
# ===================================================================

class TestBatchProcessing:
    """All frames in a batch must be processed correctly."""

    def test_batch_output_shape(self, node, batch_image):
        result = node.apply(batch_image, intensity=1.0, red_offset=1.0, green_offset=0.0, blue_offset=-1.0)
        output = result[0]
        assert output.shape == batch_image.shape

    def test_batch_output_range(self, node, batch_image):
        result = node.apply(batch_image, intensity=3.0, red_offset=3.0, green_offset=-1.0, blue_offset=-3.0)
        output = result[0]
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ===================================================================
# 6. Output dtype float32
# ===================================================================

class TestOutputDtype:
    """Return tensor must be float32."""

    def test_dtype_float32(self, node, single_image):
        result = node.apply(single_image, intensity=1.0, red_offset=1.0, green_offset=0.0, blue_offset=-1.0)
        output = result[0]
        assert output.dtype == torch.float32


# ===================================================================
# 7. CPU path works without moderngl
# ===================================================================

class TestCPUPath:
    """Node with use_gpu=False must still produce valid output."""

    def test_cpu_path_produces_output(self, node, single_image):
        assert node.use_gpu is False
        result = node.apply(single_image, intensity=2.0, red_offset=2.0, green_offset=1.0, blue_offset=-2.0)
        output = result[0]
        assert output.shape == single_image.shape
        assert output.dtype == torch.float32


# ===================================================================
# 8. SCIPY_AVAILABLE flag exists and is bool
# ===================================================================

class TestScipyFlag:
    """Module-level scipy availability flag must be a boolean."""

    def test_flag_is_bool(self):
        assert isinstance(SCIPY_AVAILABLE, bool)
