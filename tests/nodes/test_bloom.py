"""Tests for nodes.effects.bloom -- BloomShaderNode CPU fallback and basic contracts."""

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

from nodes.effects.bloom import BloomShaderNode, MODERNGL_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    """BloomShaderNode forced to CPU path."""
    n = BloomShaderNode()
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


@pytest.fixture
def dark_image():
    """Very dark image -- all values near zero [1, 64, 64, 3]."""
    return torch.full((1, 64, 64, 3), 0.05, dtype=torch.float32)


# ===================================================================
# 1. CPU fallback: correct output shape
# ===================================================================

class TestCPUFallbackShape:
    """CPU path must preserve spatial dimensions and channel count."""

    def test_output_shape_matches_input(self, node, single_image):
        result = node.apply_bloom(single_image, threshold=0.8, intensity=1.0, radius=0.5)
        output = result[0]
        assert output.shape == single_image.shape


# ===================================================================
# 2. CPU fallback: output in [0, 1]
# ===================================================================

class TestCPUFallbackRange:
    """Output pixel values must be clamped to the valid range."""

    def test_output_within_unit_range(self, node, single_image):
        result = node.apply_bloom(single_image, threshold=0.5, intensity=2.0, radius=0.8)
        output = result[0]
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ===================================================================
# 3. Zero intensity: output approximately equals input
# ===================================================================

class TestZeroIntensity:
    """When intensity=0, bloom adds nothing -- output should match input."""

    def test_zero_intensity_identity(self, node, single_image):
        result = node.apply_bloom(single_image, threshold=0.8, intensity=0.0, radius=0.5)
        output = result[0]
        assert torch.allclose(output, single_image, atol=1e-5)


# ===================================================================
# 4. High threshold on dark image: output approximately equals input
# ===================================================================

class TestHighThresholdDarkImage:
    """Threshold 0.99 on a dark image -- nothing passes, no bloom added."""

    def test_high_threshold_dark_image_identity(self, node, dark_image):
        result = node.apply_bloom(dark_image, threshold=0.99, intensity=1.0, radius=0.5)
        output = result[0]
        assert torch.allclose(output, dark_image, atol=1e-5)


# ===================================================================
# 5. Batch processing
# ===================================================================

class TestBatchProcessing:
    """All frames in a batch must be processed with correct shape."""

    def test_batch_output_shape(self, node, batch_image):
        result = node.apply_bloom(batch_image, threshold=0.8, intensity=1.0, radius=0.5)
        output = result[0]
        assert output.shape == batch_image.shape

    def test_batch_output_range(self, node, batch_image):
        result = node.apply_bloom(batch_image, threshold=0.5, intensity=2.0, radius=0.8)
        output = result[0]
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ===================================================================
# 6. Output dtype is float32
# ===================================================================

class TestOutputDtype:
    """BloomShaderNode must return float32 tensors."""

    def test_dtype_float32(self, node, single_image):
        result = node.apply_bloom(single_image, threshold=0.8, intensity=1.0, radius=0.5)
        output = result[0]
        assert output.dtype == torch.float32


# ===================================================================
# 7. MODERNGL_AVAILABLE flag exists and is bool
# ===================================================================

class TestModernGLFlag:
    """Module-level flag for GPU availability must be a boolean."""

    def test_flag_is_bool(self):
        assert isinstance(MODERNGL_AVAILABLE, bool)
