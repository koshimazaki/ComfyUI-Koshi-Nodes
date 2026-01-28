"""Tests for nodes.image.dither.nodes -- KoshiDither universal dithering node."""

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

from nodes.image.dither.nodes import KoshiDither


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    return KoshiDither()


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
def color_image():
    """Image with distinctly different RGB channels."""
    img = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    img[..., 0] = 0.9  # red channel high
    img[..., 1] = 0.1  # green channel low
    img[..., 2] = 0.5  # blue channel mid
    return img


# ===================================================================
# 1. Each technique produces correct output shape [B, H, W, 3]
# ===================================================================

class TestOutputShape:

    @pytest.mark.parametrize("technique", KoshiDither.TECHNIQUES)
    def test_shape_matches_input(self, node, sample_image, technique):
        result = node.dither(sample_image, technique, levels=4, grayscale=True)
        output = result[0]
        assert output.shape == (1, 64, 64, 3)


# ===================================================================
# 2. Output dtype is float32
# ===================================================================

class TestOutputDtype:

    @pytest.mark.parametrize("technique", KoshiDither.TECHNIQUES)
    def test_dtype_float32(self, node, sample_image, technique):
        result = node.dither(sample_image, technique, levels=4, grayscale=True)
        output = result[0]
        assert output.dtype == torch.float32


# ===================================================================
# 3. Output values in [0, 1] range
# ===================================================================

class TestOutputRange:

    @pytest.mark.parametrize("technique", KoshiDither.TECHNIQUES)
    def test_values_in_unit_range(self, node, sample_image, technique):
        result = node.dither(sample_image, technique, levels=4, grayscale=True)
        output = result[0]
        assert output.min() >= 0.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6


# ===================================================================
# 4. Floyd-Steinberg energy conservation
# ===================================================================

class TestFloydSteinbergEnergy:

    def test_mean_output_approximates_mean_input(self, node, gradient_image):
        """Floyd-Steinberg error diffusion should preserve average brightness."""
        result = node.dither(gradient_image, "floyd_steinberg", levels=4, grayscale=True)
        output = result[0]
        input_mean = gradient_image.mean().item()
        output_mean = output.mean().item()
        assert abs(input_mean - output_mean) < 0.15, (
            f"Energy not conserved: input mean={input_mean:.4f}, "
            f"output mean={output_mean:.4f}"
        )


# ===================================================================
# 5. Grayscale mode produces identical R=G=B channels
# ===================================================================

class TestGrayscaleMode:

    @pytest.mark.parametrize("technique", ["bayer", "floyd_steinberg", "atkinson", "none"])
    def test_grayscale_channels_equal(self, node, sample_image, technique):
        result = node.dither(sample_image, technique, levels=4, grayscale=True)
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        b = output[0, :, :, 2]
        assert torch.allclose(r, g), "R and G channels differ in grayscale mode"
        assert torch.allclose(g, b), "G and B channels differ in grayscale mode"


# ===================================================================
# 6. Color mode can produce different RGB channels
# ===================================================================

class TestColorMode:

    def test_color_channels_can_differ(self, node, color_image):
        """When grayscale=False, channels are dithered independently."""
        result = node.dither(color_image, "bayer", levels=4, grayscale=False)
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        # With R=0.9, G=0.1, they should quantize to different values
        assert not torch.allclose(r, g), (
            "Expected different channels in color mode with different input channels"
        )


# ===================================================================
# 7. Batch processing: 4 images processed correctly
# ===================================================================

class TestBatchProcessing:

    def test_batch_output_shape(self, node, batch_image):
        result = node.dither(batch_image, "bayer", levels=4, grayscale=True)
        output = result[0]
        assert output.shape == (4, 64, 64, 3)

    def test_batch_all_valid(self, node, batch_image):
        result = node.dither(batch_image, "floyd_steinberg", levels=4, grayscale=True)
        output = result[0]
        assert output.min() >= 0.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6


# ===================================================================
# 8. Bayer sizes: 2x2, 4x4, 8x8, 16x16 all work
# ===================================================================

class TestBayerSizes:

    @pytest.mark.parametrize("bayer_size", ["2x2", "4x4", "8x8", "16x16"])
    def test_bayer_size_runs(self, node, sample_image, bayer_size):
        result = node.dither(
            sample_image, "bayer", levels=4, grayscale=True,
            bayer_size=bayer_size,
        )
        output = result[0]
        assert output.shape == (1, 64, 64, 3)
        assert output.dtype == torch.float32


# ===================================================================
# 9. 2-level dithering produces only 0.0 and 1.0 values
# ===================================================================

class TestBinaryDithering:

    @pytest.mark.parametrize("technique", ["bayer", "floyd_steinberg", "atkinson", "none"])
    def test_two_levels_binary_output(self, node, sample_image, technique):
        result = node.dither(sample_image, technique, levels=2, grayscale=True)
        output = result[0]
        unique_vals = torch.unique(output)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5), (
                f"Expected only 0.0 or 1.0, got {val.item()}"
            )


# ===================================================================
# 10. Halftone dot shapes: circle, square, diamond
# ===================================================================

class TestHalftoneShapes:

    @pytest.mark.parametrize("dot_shape", ["circle", "square", "diamond"])
    def test_halftone_shape_runs(self, node, sample_image, dot_shape):
        result = node.dither(
            sample_image, "halftone", levels=2, grayscale=True,
            dot_size=4.0, dot_angle=45.0, dot_shape=dot_shape,
        )
        output = result[0]
        assert output.shape == (1, 64, 64, 3)
        assert output.min() >= 0.0 - 1e-6
        assert output.max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("dot_shape", ["circle", "square", "diamond"])
    def test_halftone_grayscale_channels_equal(self, node, sample_image, dot_shape):
        """Halftone always runs in grayscale mode internally."""
        result = node.dither(
            sample_image, "halftone", levels=2, grayscale=True,
            dot_size=4.0, dot_angle=45.0, dot_shape=dot_shape,
        )
        output = result[0]
        r = output[0, :, :, 0]
        g = output[0, :, :, 1]
        b = output[0, :, :, 2]
        assert torch.allclose(r, g)
        assert torch.allclose(g, b)


# ===================================================================
# 11. "none" technique just quantizes
# ===================================================================

class TestNoneTechnique:

    def test_none_quantizes_to_correct_levels(self, node, gradient_image):
        """With levels=4, 'none' should produce values from {0, 1/3, 2/3, 1}."""
        result = node.dither(gradient_image, "none", levels=4, grayscale=True)
        output = result[0]
        unique_vals = torch.unique(output)
        expected = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
        for val in unique_vals:
            closest = min(expected, key=lambda e: abs(e - val.item()))
            assert val.item() == pytest.approx(closest, abs=1e-4), (
                f"Unexpected quantized value: {val.item()}"
            )

    def test_none_does_not_diffuse_error(self, node, sample_image):
        """'none' is pure quantization -- same input pixel always maps the same."""
        result = node.dither(sample_image, "none", levels=4, grayscale=True)
        output = result[0]
        # Running twice should produce identical output
        result2 = node.dither(sample_image, "none", levels=4, grayscale=True)
        output2 = result2[0]
        assert torch.allclose(output, output2)
