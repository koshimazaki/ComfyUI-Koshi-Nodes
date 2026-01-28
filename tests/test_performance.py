"""Performance baseline tests with generous thresholds for CI.

All tests are marked @pytest.mark.slow and excluded from default CI runs.
Run explicitly with: pytest tests/test_performance.py -m slow
"""

import sys
import os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import time

import pytest
import torch
import numpy as np

from nodes.image.dither.nodes import KoshiDither
from nodes.image.binary.nodes import KoshiBinary
from nodes.image.greyscale.nodes import KoshiGreyscale
from nodes.effects.bloom import BloomShaderNode
from nodes.generators.glitch_candies import KoshiGlitchCandies
from nodes.flux_motion.core.transforms import apply_composite_transform
from nodes.flux_motion.core.schedule_parser import parse_schedule_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(height: int, width: int, batch: int = 1) -> torch.Tensor:
    """Create a reproducible test image [B, H, W, 3] float32."""
    torch.manual_seed(42)
    return torch.rand(batch, height, width, 3, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Floyd-Steinberg dithering 512x512 < 30 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFloydSteinbergPerformance:

    def test_floyd_steinberg_512x512(self):
        """Floyd-Steinberg error diffusion on 512x512 within time budget."""
        node = KoshiDither()
        image = _make_image(512, 512)

        start = time.perf_counter()
        node.dither(image, technique="floyd_steinberg", levels=2, grayscale=True)
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, (
            f"Floyd-Steinberg 512x512 took {elapsed:.2f}s, budget is 30s"
        )


# ---------------------------------------------------------------------------
# 2. Bayer dithering 512x512 < 2 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBayerPerformance:

    def test_bayer_512x512(self):
        """Bayer ordered dithering on 512x512 within time budget."""
        node = KoshiDither()
        image = _make_image(512, 512)

        start = time.perf_counter()
        node.dither(
            image, technique="bayer", levels=4, grayscale=True,
            bayer_size="8x8",
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"Bayer 512x512 took {elapsed:.2f}s, budget is 2s"
        )


# ---------------------------------------------------------------------------
# 3. 100 affine transforms (64x64 latent) < 10 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAffineTransformPerformance:

    def test_100_transforms_64x64(self):
        """100 sequential affine transforms on 64x64 latent within budget."""
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64, dtype=torch.float32)

        motion_params = {
            "zoom": 1.02,
            "angle": 1.5,
            "translation_x": 2.0,
            "translation_y": -1.0,
        }

        start = time.perf_counter()
        current = latent
        for _ in range(100):
            current = apply_composite_transform(current, motion_params)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"100 affine transforms took {elapsed:.2f}s, budget is 10s"
        )


# ---------------------------------------------------------------------------
# 4. Schedule parsing 10K frames < 5 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestScheduleParsingPerformance:

    def test_parse_10k_frames(self):
        """Parsing a complex schedule string for 10,000 frames within budget."""
        # Build a schedule with 20 keyframes spread across 10K frames
        keyframes = [f"{i * 500}:({1.0 + 0.01 * i})" for i in range(20)]
        schedule_string = ", ".join(keyframes)

        start = time.perf_counter()
        result = parse_schedule_string(
            schedule_string, num_frames=10000,
            default=1.0, interpolation="linear",
        )
        elapsed = time.perf_counter() - start

        assert len(result) == 10000
        assert elapsed < 5.0, (
            f"Schedule parsing 10K frames took {elapsed:.2f}s, budget is 5s"
        )


# ---------------------------------------------------------------------------
# 5. GlitchCandies 512x512 single frame < 10 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGlitchCandiesPerformance:

    def test_glitch_candies_512x512(self):
        """GlitchCandies pattern generation 512x512 within time budget."""
        node = KoshiGlitchCandies()

        start = time.perf_counter()
        node.generate(
            width=512, height=512, pattern="glitch_candies",
            time=0.0, scale=1.0, seed=42,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"GlitchCandies 512x512 took {elapsed:.2f}s, budget is 10s"
        )


# ---------------------------------------------------------------------------
# 6. Greyscale conversion 512x512 < 2 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGreyscalePerformance:

    def test_greyscale_512x512(self):
        """Greyscale luminosity conversion 512x512 within time budget."""
        node = KoshiGreyscale()
        image = _make_image(512, 512)

        start = time.perf_counter()
        node.convert(
            image, algorithm="luminosity",
            bit_depth="8-bit (256)", dither="none",
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"Greyscale 512x512 took {elapsed:.2f}s, budget is 2s"
        )


# ---------------------------------------------------------------------------
# 7. Bloom CPU 256x256 < 10 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBloomPerformance:

    def test_bloom_cpu_256x256(self):
        """Bloom CPU fallback on 256x256 within time budget."""
        node = BloomShaderNode()
        node.use_gpu = False
        image = _make_image(256, 256)

        start = time.perf_counter()
        node._apply_bloom_cpu(image, threshold=0.8, intensity=1.0, radius=0.5)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"Bloom CPU 256x256 took {elapsed:.2f}s, budget is 10s"
        )


# ---------------------------------------------------------------------------
# 8. Binary Otsu 512x512 < 5 seconds
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBinaryOtsuPerformance:

    def test_otsu_512x512(self):
        """Binary Otsu threshold on 512x512 within time budget."""
        node = KoshiBinary()
        image = _make_image(512, 512)

        start = time.perf_counter()
        node.convert(
            image, method="otsu", threshold=0.5,
            invert=False, output_hex=False,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, (
            f"Binary Otsu 512x512 took {elapsed:.2f}s, budget is 5s"
        )
