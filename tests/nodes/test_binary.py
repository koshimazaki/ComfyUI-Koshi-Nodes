"""Tests for nodes.image.binary.nodes -- KoshiBinary threshold and export node."""

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
import numpy as np

from nodes.image.binary.nodes import KoshiBinary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    return KoshiBinary()


@pytest.fixture
def sample_image():
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def batch_image():
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def gradient_image():
    grad = torch.linspace(0, 1, 64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    return grad.expand(1, 64, 64, 3).clone()


@pytest.fixture
def mid_gray_image():
    """Uniform 0.5 image for threshold testing."""
    return torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)


# ===================================================================
# 1. Each method produces correct output shape [B, H, W, 3]
# ===================================================================

class TestOutputShape:

    @pytest.mark.parametrize("method", KoshiBinary.METHODS)
    def test_shape_preserved(self, node, sample_image, method):
        result = node.convert(
            sample_image, method, threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        assert output.shape == (1, 64, 64, 3)


# ===================================================================
# 2. Output is truly binary: only {0.0, 1.0} values
# ===================================================================

class TestBinaryOutput:

    @pytest.mark.parametrize("method", KoshiBinary.METHODS)
    def test_only_zero_and_one(self, node, sample_image, method):
        result = node.convert(
            sample_image, method, threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5), (
                f"Expected binary values, got {val.item()} with method={method}"
            )


# ===================================================================
# 3. Simple threshold at 0.5 on gradient: roughly half white/black
# ===================================================================

class TestSimpleThreshold:

    def test_gradient_half_split(self, node, gradient_image):
        result = node.convert(
            gradient_image, "simple", threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        white_fraction = (output > 0.5).float().mean().item()
        # Gradient goes 0..1, so roughly half should be above 0.5
        assert 0.3 < white_fraction < 0.7, (
            f"Expected ~50% white pixels, got {white_fraction * 100:.1f}%"
        )


# ===================================================================
# 4. Invert: binary = 1 - original_binary
# ===================================================================

class TestInvert:

    def test_invert_flips_values(self, node, sample_image):
        result_normal = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=False,
        )
        result_inverted = node.convert(
            sample_image, "simple", threshold=0.5, invert=True, output_hex=False,
        )
        normal_out = result_normal[0]
        inverted_out = result_inverted[0]
        expected_inv = 1.0 - normal_out
        assert torch.allclose(inverted_out, expected_inv, atol=1e-5), (
            "Inverted output should equal 1 - normal output"
        )


# ===================================================================
# 5. Otsu threshold: auto-determines threshold
# ===================================================================

class TestOtsuThreshold:

    def test_otsu_produces_binary(self, node, sample_image):
        result = node.convert(
            sample_image, "otsu", threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5)

    def test_otsu_ignores_threshold_param(self, node, sample_image):
        """Otsu computes its own threshold, so different threshold args should
        produce the same result."""
        result_low = node.convert(
            sample_image, "otsu", threshold=0.1, invert=False, output_hex=False,
        )
        result_high = node.convert(
            sample_image, "otsu", threshold=0.9, invert=False, output_hex=False,
        )
        assert torch.allclose(result_low[0], result_high[0]), (
            "Otsu should auto-compute threshold, ignoring the threshold parameter"
        )


# ===================================================================
# 6. Adaptive threshold with scipy fallback
# ===================================================================

class TestAdaptiveThreshold:

    def test_adaptive_produces_binary(self, node, sample_image):
        result = node.convert(
            sample_image, "adaptive", threshold=0.5, invert=False, output_hex=False,
            block_size=11, adaptive_c=2.0,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5)

    def test_adaptive_correct_shape(self, node, sample_image):
        result = node.convert(
            sample_image, "adaptive", threshold=0.5, invert=False, output_hex=False,
            block_size=11, adaptive_c=2.0,
        )
        assert result[0].shape == (1, 64, 64, 3)


# ===================================================================
# 7. Hex output: non-empty string when output_hex=True
# ===================================================================

class TestHexOutputPresent:

    def test_hex_string_non_empty(self, node, sample_image):
        result = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=True,
        )
        hex_str = result[1]
        assert isinstance(hex_str, str)
        assert len(hex_str) > 0


# ===================================================================
# 8. Hex output format: starts with "// " and contains "const uint8_t"
# ===================================================================

class TestHexOutputFormat:

    def test_hex_starts_with_comment(self, node, sample_image):
        result = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=True,
        )
        hex_str = result[1]
        assert hex_str.startswith("// "), (
            f"Hex output should start with '// ', got: {hex_str[:30]}"
        )

    def test_hex_contains_c_array_declaration(self, node, sample_image):
        result = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=True,
        )
        hex_str = result[1]
        assert "const uint8_t" in hex_str, (
            "Hex output should contain C array declaration 'const uint8_t'"
        )

    def test_hex_contains_hex_values(self, node, sample_image):
        result = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=True,
        )
        hex_str = result[1]
        assert "0x" in hex_str, "Hex output should contain hex values starting with 0x"


# ===================================================================
# 9. Empty hex when output_hex=False
# ===================================================================

class TestHexOutputDisabled:

    def test_hex_empty_when_disabled(self, node, sample_image):
        result = node.convert(
            sample_image, "simple", threshold=0.5, invert=False, output_hex=False,
        )
        hex_str = result[1]
        assert hex_str == "", (
            f"Expected empty hex string when output_hex=False, got: {hex_str[:50]}"
        )


# ===================================================================
# 10. Batch processing works
# ===================================================================

class TestBatchProcessing:

    def test_batch_output_shape(self, node, batch_image):
        result = node.convert(
            batch_image, "simple", threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        assert output.shape == (4, 64, 64, 3)

    def test_batch_hex_output(self, node, batch_image):
        result = node.convert(
            batch_image, "simple", threshold=0.5, invert=False, output_hex=True,
        )
        hex_str = result[1]
        # Should have hex data for each batch item, separated by double newlines
        assert isinstance(hex_str, str)
        assert len(hex_str) > 0


# ===================================================================
# 11. Output dtype float32
# ===================================================================

class TestOutputDtype:

    @pytest.mark.parametrize("method", KoshiBinary.METHODS)
    def test_dtype_is_float32(self, node, sample_image, method):
        result = node.convert(
            sample_image, method, threshold=0.5, invert=False, output_hex=False,
        )
        assert result[0].dtype == torch.float32


# ===================================================================
# 12. R=G=B in output (greyscale binary)
# ===================================================================

class TestGreyscaleChannels:

    @pytest.mark.parametrize("method", KoshiBinary.METHODS)
    def test_all_channels_equal(self, node, sample_image, method):
        result = node.convert(
            sample_image, method, threshold=0.5, invert=False, output_hex=False,
        )
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        b = output[0, :, :, 2]
        assert torch.allclose(r, g), f"{method}: R != G"
        assert torch.allclose(g, b), f"{method}: G != B"


# ===================================================================
# 13. dither_bayer produces binary output
# ===================================================================

class TestDitherBayer:

    def test_dither_bayer_is_binary(self, node, sample_image):
        result = node.convert(
            sample_image, "dither_bayer", threshold=0.5, invert=False,
            output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5)

    def test_dither_bayer_not_all_same(self, node, gradient_image):
        """On a gradient, dither_bayer should produce a mix of 0 and 1."""
        result = node.convert(
            gradient_image, "dither_bayer", threshold=0.5, invert=False,
            output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        assert len(unique_vals) == 2, "Gradient should produce both 0 and 1 values"


# ===================================================================
# 14. dither_floyd produces binary output
# ===================================================================

class TestDitherFloyd:

    def test_dither_floyd_is_binary(self, node, sample_image):
        result = node.convert(
            sample_image, "dither_floyd", threshold=0.5, invert=False,
            output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5)

    def test_dither_floyd_not_all_same(self, node, gradient_image):
        """On a gradient, dither_floyd should produce a mix of 0 and 1."""
        result = node.convert(
            gradient_image, "dither_floyd", threshold=0.5, invert=False,
            output_hex=False,
        )
        output = result[0]
        unique_vals = torch.unique(output)
        assert len(unique_vals) == 2, "Gradient should produce both 0 and 1 values"
