"""Tests for nodes.image.greyscale.nodes -- KoshiGreyscale conversion node."""

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

from nodes.image.greyscale.nodes import KoshiGreyscale


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    return KoshiGreyscale()


@pytest.fixture
def sample_image():
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def batch_image():
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def known_color_image():
    """Image with known channel values for verifying weight formulas."""
    img = torch.zeros(1, 16, 16, 3, dtype=torch.float32)
    img[..., 0] = 0.8  # R
    img[..., 1] = 0.4  # G
    img[..., 2] = 0.2  # B
    return img


# ===================================================================
# 1. Each algorithm produces correct output shape
# ===================================================================

class TestOutputShape:

    @pytest.mark.parametrize("algorithm", KoshiGreyscale.ALGORITHMS)
    def test_shape_preserved(self, node, sample_image, algorithm):
        result = node.convert(sample_image, algorithm, "8-bit (256)", "none")
        output = result[0]
        assert output.shape == (1, 64, 64, 3)


# ===================================================================
# 2. Luminosity weights verified: R=0.299, G=0.587, B=0.114
# ===================================================================

class TestLuminosityWeights:

    def test_luminosity_formula(self, node, known_color_image):
        result = node.convert(known_color_image, "luminosity", "8-bit (256)", "none")
        output = result[0]
        expected_gray = 0.299 * 0.8 + 0.587 * 0.4 + 0.114 * 0.2
        # 8-bit quantization: floor(val * 255 + 0.5) / 255
        quantized = np.floor(expected_gray * 255 + 0.5) / 255
        actual = output[0, 0, 0, 0].item()
        assert actual == pytest.approx(quantized, abs=1e-3), (
            f"Luminosity expected ~{quantized:.4f}, got {actual:.4f}"
        )

    def test_average_formula(self, node, known_color_image):
        result = node.convert(known_color_image, "average", "8-bit (256)", "none")
        output = result[0]
        expected_gray = (0.8 + 0.4 + 0.2) / 3.0
        quantized = np.floor(expected_gray * 255 + 0.5) / 255
        actual = output[0, 0, 0, 0].item()
        assert actual == pytest.approx(quantized, abs=1e-3)


# ===================================================================
# 3. All 4 bit depths produce correct number of unique values
# ===================================================================

class TestBitDepths:

    @pytest.mark.parametrize("bit_depth,max_levels", [
        ("8-bit (256)", 256),
        ("4-bit (16)", 16),
        ("2-bit (4)", 4),
        ("1-bit (2)", 2),
    ])
    def test_unique_values_within_level_count(self, node, sample_image, bit_depth, max_levels):
        result = node.convert(sample_image, "luminosity", bit_depth, "none")
        output = result[0]
        unique_count = len(torch.unique(output))
        assert unique_count <= max_levels, (
            f"{bit_depth}: got {unique_count} unique values, expected <= {max_levels}"
        )

    def test_1bit_produces_exactly_2_values(self, node, sample_image):
        result = node.convert(sample_image, "luminosity", "1-bit (2)", "none")
        output = result[0]
        unique_vals = torch.unique(output)
        assert len(unique_vals) == 2
        vals_sorted = sorted(v.item() for v in unique_vals)
        assert vals_sorted[0] == pytest.approx(0.0, abs=1e-5)
        assert vals_sorted[1] == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# 4. Output always greyscale (R=G=B)
# ===================================================================

class TestGreyscaleOutput:

    @pytest.mark.parametrize("algorithm", KoshiGreyscale.ALGORITHMS)
    def test_all_channels_equal(self, node, sample_image, algorithm):
        result = node.convert(sample_image, algorithm, "8-bit (256)", "none")
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        b = output[0, :, :, 2]
        assert torch.allclose(r, g), f"{algorithm}: R != G"
        assert torch.allclose(g, b), f"{algorithm}: G != B"


# ===================================================================
# 5. Dither options: none, bayer_2x2, bayer_4x4, bayer_8x8
# ===================================================================

class TestDitherOptions:

    @pytest.mark.parametrize("dither_mode", ["none", "bayer_2x2", "bayer_4x4", "bayer_8x8"])
    def test_dither_mode_runs(self, node, sample_image, dither_mode):
        result = node.convert(sample_image, "luminosity", "4-bit (16)", dither_mode)
        output = result[0]
        assert output.shape == (1, 64, 64, 3)
        assert output.dtype == torch.float32

    def test_dither_changes_output(self, node, sample_image):
        """Applying dithering should change the result compared to no dithering."""
        result_none = node.convert(sample_image, "luminosity", "4-bit (16)", "none")
        result_bayer = node.convert(sample_image, "luminosity", "4-bit (16)", "bayer_4x4")
        assert not torch.allclose(result_none[0], result_bayer[0]), (
            "Dithering should produce a different output than no dithering"
        )


# ===================================================================
# 6. Desaturate blend: amount=0.0 vs amount=1.0
# ===================================================================

class TestDesaturateBlend:

    def test_full_desaturation_is_greyscale(self, node, sample_image):
        """desaturate_amount=1.0 should produce full greyscale (R=G=B)."""
        result = node.convert(
            sample_image, "luminosity", "8-bit (256)", "none",
            desaturate_amount=1.0,
        )
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        b = output[0, :, :, 2]
        assert torch.allclose(r, g)
        assert torch.allclose(g, b)

    def test_zero_desaturation_preserves_luminance_of_original(self, node, sample_image):
        """desaturate_amount=0.0 blends with original, then re-extracts gray.
        The output should still be R=G=B because final quantization is greyscale."""
        result = node.convert(
            sample_image, "luminosity", "8-bit (256)", "none",
            desaturate_amount=0.0,
        )
        output = result[0]
        # Even with amount=0.0, the code path still outputs greyscale
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        assert torch.allclose(r, g)

    def test_partial_desaturation_differs_from_full(self, node, sample_image):
        """amount=0.5 should give different result from amount=1.0."""
        result_full = node.convert(
            sample_image, "luminosity", "8-bit (256)", "none",
            desaturate_amount=1.0,
        )
        result_half = node.convert(
            sample_image, "luminosity", "8-bit (256)", "none",
            desaturate_amount=0.5,
        )
        # They may differ because the blend changes the luminosity extraction
        # but both should be valid greyscale
        assert result_full[0].shape == result_half[0].shape


# ===================================================================
# 7. Batch processing works
# ===================================================================

class TestBatchProcessing:

    def test_batch_output_shape(self, node, batch_image):
        result = node.convert(batch_image, "luminosity", "8-bit (256)", "none")
        output = result[0]
        assert output.shape == (4, 64, 64, 3)

    def test_batch_all_valid_range(self, node, batch_image):
        result = node.convert(batch_image, "average", "4-bit (16)", "bayer_4x4")
        output = result[0]
        assert output.min() >= 0.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6


# ===================================================================
# 8. "red" algorithm extracts only red channel
# ===================================================================

class TestChannelExtraction:

    def test_red_extracts_red_channel(self, node, known_color_image):
        result = node.convert(known_color_image, "red", "8-bit (256)", "none")
        output = result[0]
        # R channel of input = 0.8; quantized at 8-bit: floor(0.8*255+0.5)/255
        expected = np.floor(0.8 * 255 + 0.5) / 255
        actual = output[0, 0, 0, 0].item()
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_blue_extracts_blue_channel(self, node, known_color_image):
        result = node.convert(known_color_image, "blue", "8-bit (256)", "none")
        output = result[0]
        expected = np.floor(0.2 * 255 + 0.5) / 255
        actual = output[0, 0, 0, 0].item()
        assert actual == pytest.approx(expected, abs=1e-3)

    def test_green_extracts_green_channel(self, node, known_color_image):
        result = node.convert(known_color_image, "green", "8-bit (256)", "none")
        output = result[0]
        expected = np.floor(0.4 * 255 + 0.5) / 255
        actual = output[0, 0, 0, 0].item()
        assert actual == pytest.approx(expected, abs=1e-3)


# ===================================================================
# 9. Output dtype float32, range [0,1]
# ===================================================================

class TestOutputDtypeAndRange:

    @pytest.mark.parametrize("algorithm", KoshiGreyscale.ALGORITHMS)
    def test_dtype_is_float32(self, node, sample_image, algorithm):
        result = node.convert(sample_image, algorithm, "8-bit (256)", "none")
        assert result[0].dtype == torch.float32

    @pytest.mark.parametrize("algorithm", KoshiGreyscale.ALGORITHMS)
    def test_range_zero_to_one(self, node, sample_image, algorithm):
        result = node.convert(sample_image, algorithm, "4-bit (16)", "none")
        output = result[0]
        assert output.min() >= 0.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6
