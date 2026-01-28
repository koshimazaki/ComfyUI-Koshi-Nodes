"""Tests for nodes.effects.glitch -- GlitchShaderNode contracts and validation."""

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

from nodes.effects.glitch import GlitchShaderNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    """Fresh GlitchShaderNode instance."""
    return GlitchShaderNode()


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
# Default glitch parameters
# ---------------------------------------------------------------------------

GLITCH_DEFAULTS = dict(
    time=1.0,
    glitch_intensity=0.5,
    rgb_shift=6.0,
    shake_amount=8.0,
    noise_amount=0.15,
    block_noise_size=3.0,
    scan_line_intensity=0.15,
    freeze=False,
    seed=42,
)


# ===================================================================
# 1. Output shape matches input
# ===================================================================

class TestOutputShape:
    """Glitch effect must preserve tensor dimensions."""

    def test_shape_preserved(self, node, single_image):
        result = node.apply_glitch(single_image, **GLITCH_DEFAULTS)
        output = result[0]
        assert output.shape == single_image.shape


# ===================================================================
# 2. Output in [0, 1]
# ===================================================================

class TestOutputRange:
    """Output pixel values must be clamped to valid range."""

    def test_output_within_unit_range(self, node, single_image):
        result = node.apply_glitch(single_image, **GLITCH_DEFAULTS)
        output = result[0]
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6


# ===================================================================
# 3. Input validation: non-4D tensor raises ValueError
# ===================================================================

class TestInputValidation:
    """Node must reject tensors that are not 4-dimensional."""

    def test_3d_tensor_raises(self, node):
        bad_input = torch.rand(64, 64, 3, dtype=torch.float32)
        with pytest.raises(ValueError, match="4D tensor"):
            node.apply_glitch(bad_input, **GLITCH_DEFAULTS)

    def test_2d_tensor_raises(self, node):
        bad_input = torch.rand(64, 64, dtype=torch.float32)
        with pytest.raises(ValueError, match="4D tensor"):
            node.apply_glitch(bad_input, **GLITCH_DEFAULTS)

    def test_5d_tensor_raises(self, node):
        bad_input = torch.rand(1, 1, 64, 64, 3, dtype=torch.float32)
        with pytest.raises(ValueError, match="4D tensor"):
            node.apply_glitch(bad_input, **GLITCH_DEFAULTS)


# ===================================================================
# 4. Seed reproducibility: same seed produces same output
# ===================================================================

class TestSeedReproducibility:
    """Identical seed and parameters must yield identical results."""

    def test_deterministic_output(self, single_image):
        node_a = GlitchShaderNode()
        node_b = GlitchShaderNode()
        result_a = node_a.apply_glitch(single_image, **GLITCH_DEFAULTS)
        result_b = node_b.apply_glitch(single_image, **GLITCH_DEFAULTS)
        assert torch.allclose(result_a[0], result_b[0], atol=1e-6)


# ===================================================================
# 5. Freeze mode: frozen_time is preserved
# ===================================================================

class TestFreezeMode:
    """When freeze=True, the node should use frozen_time instead of current time."""

    def test_freeze_uses_stored_time(self, node, single_image):
        # First call without freeze to set frozen_time
        params = dict(GLITCH_DEFAULTS)
        params["time"] = 5.0
        params["freeze"] = False
        node.apply_glitch(single_image, **params)
        stored_time = node.frozen_time
        assert stored_time == 5.0

        # Second call with freeze=True and different time value
        params["time"] = 99.0
        params["freeze"] = True
        node.apply_glitch(single_image, **params)
        # frozen_time should remain 5.0, not change to 99.0
        assert node.frozen_time == stored_time


# ===================================================================
# 6. Batch processing
# ===================================================================

class TestBatchProcessing:
    """All frames in a batch must be processed with correct shape."""

    def test_batch_output_shape(self, node, batch_image):
        result = node.apply_glitch(batch_image, **GLITCH_DEFAULTS)
        output = result[0]
        assert output.shape == batch_image.shape

    def test_batch_output_range(self, node, batch_image):
        result = node.apply_glitch(batch_image, **GLITCH_DEFAULTS)
        output = result[0]
        assert output.min().item() >= 0.0 - 1e-6
        assert output.max().item() <= 1.0 + 1e-6


# ===================================================================
# 7. Zero intensity: output still has valid shape
# ===================================================================

class TestZeroIntensity:
    """Even with zero glitch intensity, the node must return a valid tensor."""

    def test_zero_intensity_valid_shape(self, node, single_image):
        params = dict(GLITCH_DEFAULTS)
        params["glitch_intensity"] = 0.0
        params["rgb_shift"] = 0.0
        params["shake_amount"] = 0.0
        params["noise_amount"] = 0.0
        params["scan_line_intensity"] = 0.0
        result = node.apply_glitch(single_image, **params)
        output = result[0]
        assert output.shape == single_image.shape


# ===================================================================
# 8. Output dtype float32
# ===================================================================

class TestOutputDtype:
    """GlitchShaderNode must return float32 tensors."""

    def test_dtype_float32(self, node, single_image):
        result = node.apply_glitch(single_image, **GLITCH_DEFAULTS)
        output = result[0]
        assert output.dtype == torch.float32
