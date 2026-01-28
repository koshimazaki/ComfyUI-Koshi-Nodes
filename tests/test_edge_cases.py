"""Edge cases and error handling tests for Koshi nodes.

Validates that nodes handle degenerate inputs gracefully: tiny images,
extreme parameters, missing optional dependencies, and malformed tensors.
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(batch=1, height=64, width=64, channels=3, value=None, dtype=torch.float32):
    """Create a ComfyUI-format image tensor [B, H, W, C]."""
    if value is not None:
        return torch.full((batch, height, width, channels), value, dtype=dtype)
    torch.manual_seed(42)
    return torch.rand(batch, height, width, channels, dtype=dtype)


def _make_latent(batch=1, channels=4, height=8, width=8):
    """Create a ComfyUI-format latent dict."""
    torch.manual_seed(42)
    return {"samples": torch.randn(batch, channels, height, width, dtype=torch.float32)}


# ===========================================================================
# Image edge cases
# ===========================================================================

class TestDitherEdgeCases:
    """Edge cases for KoshiDither node."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from nodes.image.dither.nodes import KoshiDither
        self.node = KoshiDither()

    def test_1x1_image(self):
        """1x1 image must not crash and return valid shape."""
        img = _make_image(1, 1, 1, 3)
        result = self.node.dither(img, "bayer", 2, True)
        assert result[0].shape == (1, 1, 1, 3)
        assert result[0].dtype == torch.float32

    def test_large_batch(self):
        """Batch of 8 images must produce 8 outputs."""
        img = _make_image(8, 16, 16, 3)
        result = self.node.dither(img, "floyd_steinberg", 2, True)
        assert result[0].shape[0] == 8

    def test_single_batch(self):
        """Single image batch (B=1) works for all techniques."""
        img = _make_image(1, 16, 16, 3)
        for technique in ["bayer", "floyd_steinberg", "atkinson", "halftone", "none"]:
            result = self.node.dither(img, technique, 2, True)
            assert result[0].shape == (1, 16, 16, 3)

    def test_very_small_image_8x8(self):
        """8x8 image through dither must not crash."""
        img = _make_image(1, 8, 8, 3)
        result = self.node.dither(img, "bayer", 4, True, bayer_size="8x8")
        assert result[0].shape == (1, 8, 8, 3)


class TestGreyscaleEdgeCases:
    """Edge cases for KoshiGreyscale node."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from nodes.image.greyscale.nodes import KoshiGreyscale
        self.node = KoshiGreyscale()

    def test_1x1_image(self):
        """1x1 image must produce valid greyscale output."""
        img = _make_image(1, 1, 1, 3)
        result = self.node.convert(img, "luminosity", "8-bit (256)", "none")
        assert result[0].shape == (1, 1, 1, 3)
        assert result[0].dtype == torch.float32

    def test_very_small_image_8x8(self):
        """8x8 image through all algorithms."""
        img = _make_image(1, 8, 8, 3)
        for algo in ["luminosity", "average", "lightness", "red", "green", "blue"]:
            result = self.node.convert(img, algo, "4-bit (16)", "none")
            assert result[0].shape == (1, 8, 8, 3)


class TestBloomEdgeCases:
    """Edge cases for BloomShaderNode (CPU path)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from nodes.effects.bloom import BloomShaderNode
        self.node = BloomShaderNode()
        # Force CPU path for deterministic testing
        self.node.use_gpu = False

    def test_all_zeros_image(self):
        """All-zeros image must not crash and remain bounded."""
        img = _make_image(1, 32, 32, 3, value=0.0)
        result = self.node.apply_bloom(img, threshold=0.8, intensity=1.0, radius=0.5)
        assert result[0].shape == (1, 32, 32, 3)
        assert torch.all(result[0] >= 0.0)
        assert torch.all(result[0] <= 1.0)

    def test_all_ones_image(self):
        """All-ones image must not crash and remain bounded."""
        img = _make_image(1, 32, 32, 3, value=1.0)
        result = self.node.apply_bloom(img, threshold=0.8, intensity=1.0, radius=0.5)
        assert result[0].shape == (1, 32, 32, 3)
        assert torch.all(result[0] >= 0.0)
        assert torch.all(result[0] <= 1.0)

    def test_bloom_without_moderngl(self):
        """CPU bloom fallback produces valid output with correct shape."""
        img = _make_image(1, 32, 32, 3)
        self.node.use_gpu = False
        result = self.node.apply_bloom(img, threshold=0.5, intensity=1.5, radius=0.3)
        assert result[0].shape == (1, 32, 32, 3)
        assert result[0].dtype == torch.float32
        assert torch.all(result[0] >= 0.0)
        assert torch.all(result[0] <= 1.0)


class TestMotionEngineEdgeCases:
    """Edge cases for KoshiMotionEngine transforms."""

    def test_extreme_zoom(self):
        """Extreme zoom (max=2.0) does not crash."""
        from nodes.flux_motion.core.transforms import apply_composite_transform
        latent = torch.randn(1, 4, 8, 8, dtype=torch.float32)
        params = {"zoom": 2.0, "angle": 0.0, "translation_x": 0.0, "translation_y": 0.0}
        result = apply_composite_transform(latent, params)
        assert result.shape == latent.shape
        assert torch.isfinite(result).all()

    def test_extreme_rotation(self):
        """Rotation at 180 degrees does not crash."""
        from nodes.flux_motion.core.transforms import apply_composite_transform
        latent = torch.randn(1, 4, 8, 8, dtype=torch.float32)
        params = {"zoom": 1.0, "angle": 180.0, "translation_x": 0.0, "translation_y": 0.0}
        result = apply_composite_transform(latent, params)
        assert result.shape == latent.shape
        assert torch.isfinite(result).all()

    def test_float16_input_to_transform(self):
        """Float16 latent input should still work or produce finite output."""
        from nodes.flux_motion.core.transforms import apply_composite_transform
        latent = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        params = {"zoom": 1.1, "angle": 10.0, "translation_x": 5.0, "translation_y": -3.0}
        result = apply_composite_transform(latent, params)
        assert result.shape == latent.shape
        assert torch.isfinite(result).all()


# ===========================================================================
# Graceful degradation (missing optional dependencies)
# ===========================================================================

class TestGracefulDegradation:
    """Nodes must degrade gracefully when optional deps are missing."""

    def test_chromatic_aberration_without_moderngl(self):
        """Chromatic aberration CPU fallback still produces output."""
        from nodes.effects.chromatic_aberration import KoshiChromaticAberration
        node = KoshiChromaticAberration()
        node.use_gpu = False
        img = _make_image(1, 32, 32, 3)
        result = node.apply(img, intensity=1.0, red_offset=1.0, green_offset=0.0, blue_offset=-1.0)
        assert result[0].shape == (1, 32, 32, 3)
        assert result[0].dtype == torch.float32

    def test_hologram_edge_detection_fallback(self):
        """Hologram edge detection works with or without scipy."""
        from nodes.effects.hologram import KoshiHologram
        node = KoshiHologram()
        img = _make_image(1, 32, 32, 3)
        result = node.apply(
            img,
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
        assert result[0].shape == (1, 32, 32, 3)
        assert result[0].dtype == torch.float32

    def test_binary_adaptive_fallback(self):
        """Binary adaptive threshold works (cumsum fallback if no scipy)."""
        from nodes.image.binary.nodes import KoshiBinary
        node = KoshiBinary()
        img = _make_image(1, 32, 32, 3)
        result = node.convert(
            img, method="adaptive", threshold=0.5, invert=False, output_hex=False,
            block_size=11, adaptive_c=2.0,
        )
        assert result[0].shape == (1, 32, 32, 3)
        # Binary output should contain only 0s and 1s
        unique_vals = torch.unique(result[0])
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_pixel_scaler_without_pil(self):
        """PixelScaler returns input unchanged when PIL is not available."""
        from nodes.export.oled_screen import KoshiPixelScaler, PIL_AVAILABLE
        node = KoshiPixelScaler()
        img = _make_image(1, 64, 64, 3)
        if not PIL_AVAILABLE:
            result = node.scale(
                img, "SSD1306 128x64", 128, 64, "lanczos", True, "black"
            )
            assert torch.equal(result[0], img)
        else:
            # PIL is available -- just verify no crash
            result = node.scale(
                img, "SSD1306 128x64", 128, 64, "lanczos", True, "black"
            )
            assert result[0].ndim == 4
            assert result[0].dtype == torch.float32

    def test_oled_screen_without_pil(self):
        """OLEDScreen returns input unchanged when PIL is not available."""
        from nodes.export.oled_screen import KoshiOLEDScreen, PIL_AVAILABLE
        node = KoshiOLEDScreen()
        img = _make_image(1, 64, 64, 3)
        if not PIL_AVAILABLE:
            result = node.view(
                img, "SSD1306 128x64", 128, 64, resize_to_screen=True,
            )
            assert torch.equal(result[0], img)
        else:
            result = node.view(
                img, "SSD1306 128x64", 128, 64, resize_to_screen=True,
            )
            assert result[0].ndim == 4
            assert result[0].dtype == torch.float32


# ===========================================================================
# Error handling
# ===========================================================================

class TestGlitchErrorHandling:
    """GlitchShaderNode input validation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from nodes.effects.glitch import GlitchShaderNode
        self.node = GlitchShaderNode()
        self.defaults = dict(
            time=0.0, glitch_intensity=0.5, rgb_shift=6.0, shake_amount=8.0,
            noise_amount=0.15, block_noise_size=3.0, scan_line_intensity=0.15,
            freeze=False, seed=0,
        )

    def test_none_image_raises(self):
        """None image must raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid"):
            self.node.apply_glitch(image=None, **self.defaults)

    def test_3d_tensor_raises(self):
        """3D tensor (missing batch dim) must raise ValueError."""
        bad = torch.rand(64, 64, 3)
        with pytest.raises(ValueError, match="must be 4D"):
            self.node.apply_glitch(image=bad, **self.defaults)

    def test_empty_batch_raises(self):
        """Empty batch (B=0) must raise ValueError."""
        bad = torch.rand(0, 64, 64, 3)
        with pytest.raises(ValueError, match="cannot be empty"):
            self.node.apply_glitch(image=bad, **self.defaults)

    def test_valid_input_succeeds(self):
        """Valid 4D tensor should not raise."""
        img = _make_image(1, 32, 32, 3)
        result = self.node.apply_glitch(image=img, **self.defaults)
        assert result[0].shape == (1, 32, 32, 3)


# ===========================================================================
# Batch edge cases
# ===========================================================================

class TestBatchEdgeCases:
    """Batch processing edge cases across multiple node types."""

    def test_single_batch_through_all_image_nodes(self):
        """B=1 image through dither, greyscale, binary without crash."""
        img = _make_image(1, 16, 16, 3)

        from nodes.image.dither.nodes import KoshiDither
        dither = KoshiDither()
        result = dither.dither(img, "bayer", 2, True)
        assert result[0].shape[0] == 1

        from nodes.image.greyscale.nodes import KoshiGreyscale
        grey = KoshiGreyscale()
        result = grey.convert(img, "luminosity", "8-bit (256)", "none")
        assert result[0].shape[0] == 1

        from nodes.image.binary.nodes import KoshiBinary
        binary = KoshiBinary()
        result = binary.convert(img, "simple", 0.5, False, False)
        assert result[0].shape[0] == 1

    def test_shape_morph_mismatched_batch(self):
        """ShapeMorph with different batch sizes -- verify behavior."""
        from nodes.generators.glitch_candies import KoshiShapeMorph
        node = KoshiShapeMorph()
        img_a = _make_image(2, 16, 16, 3)
        img_b = _make_image(3, 16, 16, 3)
        # PyTorch broadcasting: (2,16,16,3) and (3,16,16,3) will fail
        # Verify it either raises or handles gracefully
        try:
            result = node.morph(img_a, img_b, 0.5, "linear")
            # If it succeeds (broadcasting worked somehow), check output is valid
            assert result[0].ndim == 4
        except RuntimeError:
            # Expected: batch dimension mismatch in torch broadcasting
            pass

    def test_shape_morph_matching_batch(self):
        """ShapeMorph with matching batch sizes works correctly."""
        from nodes.generators.glitch_candies import KoshiShapeMorph
        node = KoshiShapeMorph()
        img_a = _make_image(2, 16, 16, 3, value=0.0)
        img_b = _make_image(2, 16, 16, 3, value=1.0)
        result = node.morph(img_a, img_b, 0.5, "linear")
        assert result[0].shape == (2, 16, 16, 3)
        # At t=0.5, linear blend of 0 and 1 should be ~0.5
        assert torch.allclose(result[0], torch.full_like(result[0], 0.5), atol=1e-5)

    def test_hologram_batch(self):
        """Hologram with multi-image batch produces correct count."""
        from nodes.effects.hologram import KoshiHologram
        node = KoshiHologram()
        img = _make_image(4, 32, 32, 3)
        result = node.apply(
            img,
            color_preset="cyan",
            scanline_intensity=0.3,
            scanline_count=50,
            glitch_intensity=0.0,
            edge_glow=0.2,
            grid_opacity=0.1,
            grid_size=10,
            alpha=0.9,
            time=0.0,
        )
        assert result[0].shape[0] == 4
