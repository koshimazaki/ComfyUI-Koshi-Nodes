"""E2E tests for the 40→18 node consolidation.

Covers:
- KoshiEffects: all 7 effect types, stacking, and cross-node chains
- KoshiOLEDScreen: pass-through and resize execution
- KoshiSpriteSheet: grid layout from batches
- KoshiMetadata: capture, display, and save_json
- Cross-group pipelines: generators → effects → export
- Batch consistency across all nodes
"""

import json
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
import numpy as np

from nodes.effects.koshi_effects import KoshiEffects
from nodes.export.oled_screen import KoshiOLEDScreen, KoshiSpriteSheet
from nodes.utility.koshi_metadata import KoshiMetadata
from nodes.generators.glitch_candies import KoshiGlitchCandies, KoshiShapeMorph
from nodes.image.dither.nodes import KoshiDither
from nodes.image.binary.nodes import KoshiBinary
from nodes.image.greyscale.nodes import KoshiGreyscale
from nodes.flux_motion.schedule import KoshiSchedule
from nodes.flux_motion.motion_engine import KoshiMotionEngine
from nodes.flux_motion.feedback import KoshiFeedback
from tests.conftest import unwrap_output
from tests.mocks.comfyui import MockVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(batch=1, h=64, w=64):
    """Reproducible random test image [B, H, W, 3] float32."""
    torch.manual_seed(42)
    return torch.rand(batch, h, w, 3, dtype=torch.float32)


def _validate(tensor, batch, h, w, channels=3):
    """Assert valid ComfyUI IMAGE tensor."""
    assert tensor.shape == (batch, h, w, channels), f"Expected {(batch, h, w, channels)}, got {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1e-6
    assert tensor.max() <= 1.0 + 1e-6


def _is_modified(original, result):
    """Check that the effect actually changed the image."""
    return not torch.allclose(original, result, atol=1e-4)


# ===========================================================================
# 1. KoshiEffects — All 7 Effect Types
# ===========================================================================

class TestKoshiEffectsAllTypes:
    """Test each effect type produces valid, modified output."""

    @pytest.mark.parametrize("effect_type", KoshiEffects.EFFECT_TYPES)
    def test_effect_produces_valid_output(self, effect_type):
        node = KoshiEffects()
        image = _make_image()

        # Chromatic needs higher intensity since int(offset * intensity) must be non-zero
        intensity = 1.0 if effect_type == "chromatic" else 0.5
        result = node.apply(image, effect_type=effect_type, intensity=intensity, seed=42)
        output = unwrap_output(result)
        (out_image,) = output

        _validate(out_image, 1, 64, 64)
        assert _is_modified(image, out_image), f"{effect_type} did not modify the image"

    def test_dither_with_all_methods(self):
        node = KoshiEffects()
        image = _make_image()
        for method in KoshiEffects.DITHER_METHODS:
            result = node.apply(image, effect_type="dither", intensity=0.5,
                                dither_method=method, dither_levels=4, seed=42)
            (out,) = unwrap_output(result)
            _validate(out, 1, 64, 64)

    def test_hologram_color_variants(self):
        node = KoshiEffects()
        image = _make_image()
        for color in ["cyan", "green", "purple", "orange", "white"]:
            result = node.apply(image, effect_type="hologram", intensity=0.7,
                                holo_color=color, seed=42)
            (out,) = unwrap_output(result)
            _validate(out, 1, 64, 64)

    def test_zero_intensity_passthrough(self):
        """Zero intensity should return image close to original for most effects."""
        node = KoshiEffects()
        image = _make_image()
        # Scanlines at zero intensity should not modify
        result = node.apply(image, effect_type="scanlines", intensity=0.0, seed=42)
        (out,) = unwrap_output(result)
        _validate(out, 1, 64, 64)
        assert torch.allclose(image, out, atol=1e-5)


# ===========================================================================
# 2. KoshiEffects Chains — Stacking Effects
# ===========================================================================

class TestKoshiEffectsChains:

    def test_glitch_candies_to_effects_to_greyscale(self):
        """GlitchCandies → KoshiEffects(glitch) → Greyscale."""
        gen = KoshiGlitchCandies()
        fx = KoshiEffects()
        grey = KoshiGreyscale()

        gen_result = gen.generate(width=64, height=64, pattern="plasma",
                                  time=0.0, scale=1.0, seed=42)
        gen_image, _ = unwrap_output(gen_result)
        gen_image = gen_image.float()

        fx_result = fx.apply(gen_image, effect_type="glitch", intensity=0.5, seed=42)
        (fx_out,) = unwrap_output(fx_result)
        _validate(fx_out, 1, 64, 64)

        (grey_out,) = grey.convert(fx_out, algorithm="luminosity",
                                    bit_depth="8-bit (256)", dither="none")
        _validate(grey_out, 1, 64, 64)

    def test_greyscale_to_hologram_to_binary(self):
        """Greyscale → KoshiEffects(hologram) → Binary."""
        grey = KoshiGreyscale()
        fx = KoshiEffects()
        binary = KoshiBinary()

        image = _make_image()
        (grey_out,) = grey.convert(image, algorithm="luminosity",
                                    bit_depth="8-bit (256)", dither="none")

        fx_result = fx.apply(grey_out, effect_type="hologram", intensity=0.6,
                              holo_color="cyan", seed=42)
        (fx_out,) = unwrap_output(fx_result)
        _validate(fx_out, 1, 64, 64)

        binary_out, _ = binary.convert(fx_out, method="simple", threshold=0.5,
                                        invert=False, output_hex=False)
        _validate(binary_out, 1, 64, 64)

    def test_stacking_dither_then_bloom(self):
        """KoshiEffects(dither) → KoshiEffects(bloom) — stacking effects."""
        fx = KoshiEffects()
        image = _make_image()

        dither_result = fx.apply(image, effect_type="dither", intensity=0.5,
                                  dither_method="bayer", dither_levels=4, seed=42)
        (dithered,) = unwrap_output(dither_result)
        _validate(dithered, 1, 64, 64)

        bloom_result = fx.apply(dithered, effect_type="bloom", intensity=0.5,
                                 bloom_threshold=0.5, bloom_radius=0.3, seed=42)
        (bloomed,) = unwrap_output(bloom_result)
        _validate(bloomed, 1, 64, 64)


# ===========================================================================
# 3. KoshiOLEDScreen Execution
# ===========================================================================

class TestKoshiOLEDScreen:

    def test_passthrough_no_resize(self):
        """OLED screen with resize_to_screen=False passes image through."""
        node = KoshiOLEDScreen()
        image = _make_image()

        result = node.view(image, screen_preset="SSD1306 128x64",
                           custom_width=256, custom_height=128,
                           resize_to_screen=False)
        (out,) = unwrap_output(result)
        _validate(out, 1, 64, 64)
        assert torch.allclose(image, out)

    def test_resize_to_screen(self):
        """OLED screen resizes image to screen dimensions."""
        node = KoshiOLEDScreen()
        image = _make_image(h=128, w=256)

        result = node.view(image, screen_preset="SSD1306 128x64",
                           custom_width=256, custom_height=128,
                           resize_to_screen=True)
        (out,) = unwrap_output(result)
        # SSD1306 128x64 → width=128, height=64
        _validate(out, 1, 64, 128)

    def test_batch_passthrough(self):
        """Batch of images passes through OLED screen."""
        node = KoshiOLEDScreen()
        image = _make_image(batch=4)

        result = node.view(image, screen_preset="Custom",
                           custom_width=64, custom_height=64,
                           resize_to_screen=False)
        (out,) = unwrap_output(result)
        _validate(out, 4, 64, 64)


# ===========================================================================
# 4. KoshiSpriteSheet
# ===========================================================================

class TestKoshiSpriteSheet:

    def test_batch_to_grid(self):
        """4-image batch → 2x2 grid sprite sheet."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=4, h=32, w=32)

        sheet, cols, rows, count = node.create_sheet(
            images, layout="auto", max_cols=8, max_rows=8,
            padding=0, background="black"
        )

        assert count == 4
        assert cols == 2
        assert rows == 2
        # Sheet should be [1, 64, 64, 3] for 2x2 grid of 32x32
        _validate(sheet, 1, 64, 64)

    def test_horizontal_layout(self):
        """3 images → horizontal strip."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=3, h=32, w=32)

        sheet, cols, rows, count = node.create_sheet(
            images, layout="horizontal", max_cols=8, max_rows=8,
            padding=0, background="black"
        )

        assert cols == 3
        assert rows == 1
        assert count == 3
        _validate(sheet, 1, 32, 96)

    def test_with_padding(self):
        """Sprite sheet with padding between frames."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=4, h=32, w=32)

        sheet, cols, rows, count = node.create_sheet(
            images, layout="grid", max_cols=2, max_rows=2,
            padding=2, background="white"
        )

        # 2x2 grid with padding=2: (32+2)*2 - 2 = 66
        expected_w = 2 * (32 + 2) - 2  # 66
        expected_h = 2 * (32 + 2) - 2  # 66
        _validate(sheet, 1, expected_h, expected_w)

    def test_glitch_candies_batch_to_sprite(self):
        """GlitchCandies (batch=4) → SpriteSheet."""
        gen = KoshiGlitchCandies()
        sprite = KoshiSpriteSheet()

        gen_result = gen.generate(width=32, height=32, pattern="plasma",
                                   time=0.0, scale=1.0, seed=42,
                                   batch_size=4)
        gen_image, _ = unwrap_output(gen_result)
        gen_image = gen_image.float()
        assert gen_image.shape[0] == 4

        sheet, cols, rows, count = sprite.create_sheet(
            gen_image, layout="auto", max_cols=8, max_rows=8,
            padding=0, background="black"
        )
        assert count == 4
        assert sheet.shape[0] == 1  # single sheet output


# ===========================================================================
# 5. KoshiMetadata
# ===========================================================================

class TestKoshiMetadata:

    def test_capture_only_with_image(self):
        """capture_only with image produces valid JSON with image metadata."""
        node = KoshiMetadata()
        image = _make_image()
        mock_prompt = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20, "cfg": 7.0}},
        }

        metadata_json, display_text, file_path = node.process(
            action="capture_only", image=image, custom_note="test note",
            prompt=mock_prompt
        )

        data = json.loads(metadata_json)
        assert data["generator"] == "Koshi-Nodes"
        assert data["note"] == "test note"
        assert data["output_image"]["batch_size"] == 1
        assert data["output_image"]["height"] == 64
        assert data["output_image"]["width"] == 64
        assert file_path == ""  # capture_only doesn't save

    def test_capture_only_no_image(self):
        """capture_only without image doesn't crash."""
        node = KoshiMetadata()
        metadata_json, display_text, file_path = node.process(action="capture_only")

        data = json.loads(metadata_json)
        assert "output_image" not in data
        assert "timestamp" in data

    def test_save_json(self, tmp_path):
        """save_json writes JSON file to disk."""
        node = KoshiMetadata()
        input_data = json.dumps({"test": "data", "value": 42})

        metadata_json, display_text, file_path = node.process(
            action="save_json",
            input_json=input_data,
            filename="test_meta",
            output_path=str(tmp_path),
            append_timestamp=False,
        )

        assert os.path.exists(file_path)
        with open(file_path) as f:
            saved = json.load(f)
        assert saved["test"] == "data"
        assert saved["value"] == 42

    def test_display_text_formatting(self):
        """Display text contains expected sections."""
        node = KoshiMetadata()
        image = _make_image(batch=2, h=128, w=256)
        mock_prompt = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 123, "steps": 30}},
            "2": {"class_type": "CheckpointLoader", "inputs": {"ckpt_name": "model.safetensors"}},
        }

        metadata_json, display_text, _ = node.process(
            action="capture_only", image=image, prompt=mock_prompt
        )

        assert "Generation Metadata" in display_text
        assert "256x128" in display_text  # width x height
        assert "batch: 2" in display_text


# ===========================================================================
# 6. Full Cross-Group Pipelines
# ===========================================================================

class TestCrossGroupPipelines:

    def test_generator_to_effect_to_sprite(self):
        """GlitchCandies → KoshiEffects(scanlines) → SpriteSheet."""
        gen = KoshiGlitchCandies()
        fx = KoshiEffects()
        sprite = KoshiSpriteSheet()

        gen_result = gen.generate(width=32, height=32, pattern="voronoi",
                                   time=0.0, scale=1.0, seed=42, batch_size=4)
        gen_image, _ = unwrap_output(gen_result)
        gen_image = gen_image.float()

        fx_result = fx.apply(gen_image, effect_type="scanlines", intensity=0.3,
                              scanline_count=50, seed=42)
        (fx_out,) = unwrap_output(fx_result)
        _validate(fx_out, 4, 32, 32)

        sheet, cols, rows, count = sprite.create_sheet(
            fx_out, layout="auto", max_cols=8, max_rows=8,
            padding=0, background="black"
        )
        assert count == 4
        assert sheet.shape[0] == 1

    def test_greyscale_to_scanlines_to_binary(self):
        """Greyscale → KoshiEffects(scanlines) → Binary."""
        grey = KoshiGreyscale()
        fx = KoshiEffects()
        binary = KoshiBinary()

        image = _make_image()
        (grey_out,) = grey.convert(image, algorithm="luminosity",
                                    bit_depth="8-bit (256)", dither="none")

        fx_result = fx.apply(grey_out, effect_type="scanlines", intensity=0.4,
                              scanline_count=100, seed=42)
        (fx_out,) = unwrap_output(fx_result)
        _validate(fx_out, 1, 64, 64)

        binary_out, _ = binary.convert(fx_out, method="simple", threshold=0.5,
                                        invert=False, output_hex=False)
        _validate(binary_out, 1, 64, 64)

    def test_motion_schedule_to_engine(self):
        """Schedule → MotionEngine processes latent."""
        schedule = KoshiSchedule()
        engine = KoshiMotionEngine()

        schedule_result = schedule.parse(
            schedule_string="0:(1.0), 30:(1.05), 60:(1.0)",
            max_frames=60,
            interpolation="linear",
            easing="none",
        )

        # MotionEngine with direct params (no schedule feed for simplicity)
        torch.manual_seed(42)
        latent = {"samples": torch.randn(1, 4, 8, 8)}

        result = engine.process(
            latent=latent, zoom=1.05, angle=5.0,
            translation_x=2.0, translation_y=0.0,
        )

        out_latent = result[0]
        assert "samples" in out_latent
        assert out_latent["samples"].shape == (1, 4, 8, 8)

    def test_feedback_loop(self):
        """Feedback processor with MockVAE."""
        feedback = KoshiFeedback()
        vae = MockVAE()

        current = _make_image()
        torch.manual_seed(99)
        reference = torch.rand(1, 64, 64, 3, dtype=torch.float32)

        enhanced_image, encoded_latent = feedback.process(
            current_image=current,
            reference_image=reference,
            vae=vae,
            color_match_strength=0.5,
            noise_amount=0.01,
            sharpen_amount=0.1,
            contrast_boost=1.0,
        )

        _validate(enhanced_image, 1, 64, 64)
        assert "samples" in encoded_latent

    def test_shape_morph_to_effects_to_dither(self):
        """ShapeMorph → KoshiEffects(chromatic) → Dither."""
        morph = KoshiShapeMorph()
        fx = KoshiEffects()
        dither = KoshiDither()

        img_a = _make_image()
        torch.manual_seed(99)
        img_b = torch.rand(1, 64, 64, 3, dtype=torch.float32)

        morph_result = morph.morph(img_a, img_b, blend=0.5, blend_mode="linear")
        morph_image, _ = unwrap_output(morph_result)
        morph_image = morph_image.float()
        _validate(morph_image, 1, 64, 64)

        fx_result = fx.apply(morph_image, effect_type="chromatic", intensity=0.5,
                              red_offset=2.0, blue_offset=-2.0, seed=42)
        (fx_out,) = unwrap_output(fx_result)
        _validate(fx_out, 1, 64, 64)

        (dithered,) = dither.dither(fx_out, technique="bayer", levels=4,
                                     grayscale=True)
        _validate(dithered, 1, 64, 64)


# ===========================================================================
# 7. Batch Consistency
# ===========================================================================

class TestBatchConsistency:

    @pytest.mark.parametrize("effect_type", KoshiEffects.EFFECT_TYPES)
    def test_batch_through_effects(self, effect_type):
        """4-image batch through each KoshiEffects type preserves batch dim."""
        fx = KoshiEffects()
        batch = _make_image(batch=4)

        result = fx.apply(batch, effect_type=effect_type, intensity=0.5, seed=42)
        (out,) = unwrap_output(result)
        _validate(out, 4, 64, 64)

    def test_batch_full_pipeline(self):
        """Batch through generator → effect → greyscale → sprite sheet."""
        gen = KoshiGlitchCandies()
        fx = KoshiEffects()
        grey = KoshiGreyscale()
        sprite = KoshiSpriteSheet()

        gen_result = gen.generate(width=32, height=32, pattern="checkerboard",
                                   time=0.0, scale=1.0, seed=42, batch_size=4)
        gen_image, _ = unwrap_output(gen_result)
        gen_image = gen_image.float()
        assert gen_image.shape[0] == 4

        fx_result = fx.apply(gen_image, effect_type="bloom", intensity=0.3,
                              bloom_threshold=0.6, seed=42)
        (fx_out,) = unwrap_output(fx_result)
        assert fx_out.shape[0] == 4

        (grey_out,) = grey.convert(fx_out, algorithm="luminosity",
                                    bit_depth="4-bit (16)", dither="none")
        assert grey_out.shape[0] == 4

        sheet, cols, rows, count = sprite.create_sheet(
            grey_out, layout="auto", max_cols=8, max_rows=8,
            padding=0, background="black"
        )
        assert count == 4
        assert sheet.shape[0] == 1

    def test_batch_dimension_preserved_greyscale_binary(self):
        """4-image batch through Greyscale → Binary preserves batch."""
        grey = KoshiGreyscale()
        binary = KoshiBinary()

        batch = _make_image(batch=4)
        (grey_out,) = grey.convert(batch, algorithm="luminosity",
                                    bit_depth="8-bit (256)", dither="none")
        assert grey_out.shape[0] == 4

        binary_out, _ = binary.convert(grey_out, method="simple", threshold=0.5,
                                        invert=False, output_hex=False)
        assert binary_out.shape[0] == 4
        _validate(binary_out, 4, 64, 64)
