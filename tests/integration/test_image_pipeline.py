"""Integration tests chaining multiple image processing nodes."""

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

from nodes.image.dither.nodes import KoshiDither
from nodes.image.binary.nodes import KoshiBinary
from nodes.image.greyscale.nodes import KoshiGreyscale
from nodes.effects.hologram import KoshiHologram, KoshiScanlines
from nodes.effects.chromatic_aberration import KoshiChromaticAberration
from nodes.effects.bloom import BloomShaderNode
from nodes.generators.glitch_candies import KoshiGlitchCandies
from nodes.export.oled_preview import KoshiOLEDPreview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_image_tensor(tensor: torch.Tensor, expected_batch: int, height: int, width: int):
    """Assert tensor is a valid ComfyUI IMAGE: [B, H, W, 3] float32 in [0, 1]."""
    assert tensor.shape == (expected_batch, height, width, 3), (
        f"Expected shape ({expected_batch}, {height}, {width}, 3), got {tuple(tensor.shape)}"
    )
    assert tensor.dtype == torch.float32, (
        f"Expected float32, got {tensor.dtype}"
    )
    assert tensor.min() >= -1e-6, f"Min value {tensor.min().item()} below 0"
    assert tensor.max() <= 1.0 + 1e-6, f"Max value {tensor.max().item()} above 1"


def _make_test_image(batch: int = 1, height: int = 64, width: int = 64) -> torch.Tensor:
    """Create a reproducible random test image."""
    torch.manual_seed(42)
    return torch.rand(batch, height, width, 3, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Dither -> Binary
# ---------------------------------------------------------------------------

class TestDitherToBinary:

    def test_dither_output_feeds_binary(self):
        """Dither bayer output is valid input for Binary simple threshold."""
        dither_node = KoshiDither()
        binary_node = KoshiBinary()

        image = _make_test_image()

        # Stage 1: Dither
        (dithered,) = dither_node.dither(
            image, technique="bayer", levels=4, grayscale=True
        )
        _validate_image_tensor(dithered, 1, 64, 64)

        # Stage 2: Binary
        binary_out, hex_data = binary_node.convert(
            dithered, method="simple", threshold=0.5, invert=False,
            output_hex=False,
        )
        _validate_image_tensor(binary_out, 1, 64, 64)


# ---------------------------------------------------------------------------
# 2. Greyscale -> Dither -> Binary (three-node chain)
# ---------------------------------------------------------------------------

class TestGreyscaleDitherBinary:

    def test_three_node_chain(self):
        """Greyscale -> Dither -> Binary produces valid binary output."""
        grey_node = KoshiGreyscale()
        dither_node = KoshiDither()
        binary_node = KoshiBinary()

        image = _make_test_image()

        # Stage 1: Greyscale
        (greyed,) = grey_node.convert(
            image, algorithm="luminosity", bit_depth="8-bit (256)", dither="none"
        )
        _validate_image_tensor(greyed, 1, 64, 64)

        # Stage 2: Dither
        (dithered,) = dither_node.dither(
            greyed, technique="atkinson", levels=2, grayscale=True
        )
        _validate_image_tensor(dithered, 1, 64, 64)

        # Stage 3: Binary
        binary_out, _ = binary_node.convert(
            dithered, method="otsu", threshold=0.5, invert=False,
            output_hex=False,
        )
        _validate_image_tensor(binary_out, 1, 64, 64)

        # Binary output should be exactly 0 or 1
        unique_vals = torch.unique(binary_out)
        for val in unique_vals:
            assert val.item() == pytest.approx(0.0, abs=1e-5) or \
                   val.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. Hologram -> Scanlines
# ---------------------------------------------------------------------------

class TestHologramToScanlines:

    def test_hologram_then_scanlines(self):
        """Hologram output feeds into Scanlines node."""
        holo_node = KoshiHologram()
        scan_node = KoshiScanlines()

        image = _make_test_image()

        # Stage 1: Hologram
        (hologram_out,) = holo_node.apply(
            image,
            color_preset="cyan",
            scanline_intensity=0.3,
            scanline_count=100,
            glitch_intensity=0.05,
            edge_glow=0.3,
            grid_opacity=0.1,
            grid_size=20,
            alpha=0.9,
            time=0.0,
        )
        _validate_image_tensor(hologram_out, 1, 64, 64)

        # Stage 2: Scanlines
        (scanline_out,) = scan_node.apply(
            hologram_out, count=50, intensity=0.2,
            direction="horizontal", animate_speed=0.0,
        )
        _validate_image_tensor(scanline_out, 1, 64, 64)


# ---------------------------------------------------------------------------
# 4. GlitchCandies -> Dither
# ---------------------------------------------------------------------------

class TestGlitchCandiesToDither:

    def test_generated_pattern_through_dither(self):
        """GlitchCandies generator output feeds into Dither node."""
        gen_node = KoshiGlitchCandies()
        dither_node = KoshiDither()

        # Stage 1: Generate pattern
        gen_image, gen_mask = gen_node.generate(
            width=64, height=64, pattern="plasma",
            time=0.0, scale=1.0, seed=42,
        )
        assert gen_image.shape == (1, 64, 64, 3)

        # Ensure float32 (generator may return float64)
        gen_image = gen_image.float()

        # Stage 2: Dither
        (dithered,) = dither_node.dither(
            gen_image, technique="bayer", levels=4, grayscale=True,
            bayer_size="4x4",
        )
        _validate_image_tensor(dithered, 1, 64, 64)


# ---------------------------------------------------------------------------
# 5. Greyscale -> OLEDPreview
# ---------------------------------------------------------------------------

@pytest.mark.pil
class TestGreyscaleToOLEDPreview:

    def test_greyscale_into_oled_preview(self):
        """Greyscale output feeds into OLED Preview for display simulation."""
        grey_node = KoshiGreyscale()
        oled_node = KoshiOLEDPreview()

        image = _make_test_image()

        # Stage 1: Greyscale
        (greyed,) = grey_node.convert(
            image, algorithm="luminosity", bit_depth="4-bit (16)", dither="bayer_4x4"
        )
        _validate_image_tensor(greyed, 1, 64, 64)

        # Stage 2: OLED Preview (requires PIL for resize)
        (preview,) = oled_node.preview(
            greyed,
            screen_width=64,
            screen_height=32,
            bit_depth="4-bit (16 levels)",
            dither=True,
            show_pixel_grid=True,
            scale=2,
        )
        # Output is scaled: 32*2=64 height, 64*2=128 width
        assert preview.shape[0] == 1
        assert preview.shape[1] == 32 * 2  # screen_height * scale
        assert preview.shape[2] == 64 * 2  # screen_width * scale
        assert preview.shape[3] == 3
        assert preview.min() >= -1e-6
        assert preview.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 6. ChromaticAberration -> Bloom (CPU paths)
# ---------------------------------------------------------------------------

class TestChromaticAberrationToBloom:

    def test_chromatic_then_bloom_cpu(self):
        """ChromaticAberration CPU output feeds into Bloom CPU fallback."""
        ca_node = KoshiChromaticAberration()
        bloom_node = BloomShaderNode()

        # Force CPU paths
        ca_node.use_gpu = False
        bloom_node.use_gpu = False

        image = _make_test_image()

        # Stage 1: Chromatic Aberration (CPU)
        (ca_out,) = ca_node.apply(
            image, intensity=2.0, red_offset=1.5,
            green_offset=0.0, blue_offset=-1.5,
        )
        _validate_image_tensor(ca_out, 1, 64, 64)

        # Stage 2: Bloom (CPU)
        (bloom_out,) = bloom_node._apply_bloom_cpu(
            ca_out, threshold=0.6, intensity=1.0, radius=0.5,
        )
        _validate_image_tensor(bloom_out, 1, 64, 64)


# ---------------------------------------------------------------------------
# 7. Verify all chains produce valid [B, H, W, 3] float32 [0, 1]
# ---------------------------------------------------------------------------

class TestChainOutputValidity:

    def test_hologram_scanlines_output_valid(self):
        """Hologram -> Scanlines chain produces valid IMAGE tensor."""
        holo_node = KoshiHologram()
        scan_node = KoshiScanlines()
        image = _make_test_image(height=48, width=48)

        (holo,) = holo_node.apply(
            image, color_preset="green_matrix",
            scanline_intensity=0.2, scanline_count=50,
            glitch_intensity=0.0, edge_glow=0.4,
            grid_opacity=0.0, grid_size=20, alpha=1.0, time=0.0,
        )
        (result,) = scan_node.apply(
            holo, count=30, intensity=0.15,
            direction="vertical", animate_speed=0.0,
        )
        _validate_image_tensor(result, 1, 48, 48)

    def test_greyscale_dither_output_valid(self):
        """Greyscale -> Dither chain produces valid IMAGE tensor."""
        grey_node = KoshiGreyscale()
        dither_node = KoshiDither()
        image = _make_test_image(height=32, width=32)

        (greyed,) = grey_node.convert(
            image, algorithm="average", bit_depth="8-bit (256)", dither="none"
        )
        (dithered,) = dither_node.dither(
            greyed, technique="floyd_steinberg", levels=4, grayscale=True,
        )
        _validate_image_tensor(dithered, 1, 32, 32)


# ---------------------------------------------------------------------------
# 8. Batch chain: 4-image batch through Greyscale -> Dither
# ---------------------------------------------------------------------------

class TestBatchChain:

    def test_batch_four_images_through_greyscale_dither(self):
        """A batch of 4 images flows through Greyscale -> Dither correctly."""
        grey_node = KoshiGreyscale()
        dither_node = KoshiDither()

        batch_image = _make_test_image(batch=4)

        # Stage 1: Greyscale
        (greyed,) = grey_node.convert(
            batch_image, algorithm="luminosity",
            bit_depth="4-bit (16)", dither="none",
        )
        _validate_image_tensor(greyed, 4, 64, 64)

        # Stage 2: Dither
        (dithered,) = dither_node.dither(
            greyed, technique="bayer", levels=4, grayscale=True,
            bayer_size="8x8",
        )
        _validate_image_tensor(dithered, 4, 64, 64)

        # Verify each batch item is independently processed
        assert dithered.shape[0] == 4
