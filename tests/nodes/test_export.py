"""Tests for export nodes: SIDKITExport, KoshiOLEDPreview, KoshiPixelScaler,
KoshiSpriteSheet, KoshiXBMExport."""

import sys
import os
import struct

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch
import numpy as np

from nodes.export.sidkit import SIDKITExport
from nodes.export.oled_preview import KoshiOLEDPreview
from nodes.export.oled_screen import KoshiPixelScaler, KoshiSpriteSheet, KoshiXBMExport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(batch=1, h=64, w=64, c=3):
    """Create a ComfyUI-format image tensor [B, H, W, C] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(batch, h, w, c, dtype=torch.float32)


# ===================================================================
# 1. SIDKITExport
# ===================================================================

class TestSIDKITExportSIDV:
    """SIDV binary format output."""

    def test_sidv_creates_file(self, tmp_path):
        node = SIDKITExport()
        images = _make_image(batch=2, h=128, w=256)
        result = node.export_sidkit(
            images=images,
            bit_depth="1-bit (mono)",
            target_width=256,
            target_height=128,
            fps=30,
            output_format="sidv",
            filename="test_out",
            output_path=str(tmp_path),
        )
        file_path = result[0]
        assert os.path.isfile(file_path)

    def test_sidv_magic_bytes(self, tmp_path):
        """SIDV header must start with 'SIDK' magic bytes."""
        node = SIDKITExport()
        images = _make_image(batch=1, h=128, w=256)
        result = node.export_sidkit(
            images=images,
            bit_depth="1-bit (mono)",
            target_width=256,
            target_height=128,
            fps=30,
            output_format="sidv",
            filename="magic_test",
            output_path=str(tmp_path),
        )
        file_path = result[0]
        with open(file_path, "rb") as f:
            header = f.read(16)
        assert header[:4] == b"SIDK"
        assert header[4] == 1  # version

    def test_sidv_returns_frame_count(self, tmp_path):
        node = SIDKITExport()
        images = _make_image(batch=3, h=64, w=128)
        result = node.export_sidkit(
            images=images,
            bit_depth="1-bit (mono)",
            target_width=128,
            target_height=64,
            fps=24,
            output_format="sidv",
            filename="count_test",
            output_path=str(tmp_path),
        )
        assert result[1] == 3  # frame_count

    def test_sidv_file_size_positive(self, tmp_path):
        node = SIDKITExport()
        images = _make_image(batch=1, h=64, w=128)
        result = node.export_sidkit(
            images=images,
            bit_depth="4-bit (16 levels)",
            target_width=128,
            target_height=64,
            fps=30,
            output_format="sidv",
            filename="size_test",
            output_path=str(tmp_path),
        )
        assert result[2] > 16  # at least header size


class TestSIDKITExportXBM:
    """XBM text format output."""

    def test_xbm_creates_text_file(self, tmp_path):
        node = SIDKITExport()
        images = _make_image(batch=1, h=64, w=128)
        result = node.export_sidkit(
            images=images,
            bit_depth="1-bit (mono)",
            target_width=128,
            target_height=64,
            fps=30,
            output_format="xbm",
            filename="xbm_test",
            output_path=str(tmp_path),
        )
        file_path = result[0]
        assert os.path.isfile(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        assert "#define xbm_test_width 128" in content
        assert "#define xbm_test_height 64" in content
        assert "static unsigned char" in content


class TestSIDKITExportBitPacking:
    """Bit packing correctness for 1-bit, 2-bit, and 4-bit depths."""

    def test_pack_1bit_byte_count(self):
        """1-bit: 8 pixels per byte, so 256x128 -> 4096 bytes per frame."""
        node = SIDKITExport()
        frame = np.ones((128, 256), dtype=np.uint8)
        packed = node._pack_1bit(frame)
        expected_bytes = 128 * (256 // 8)
        assert len(packed) == expected_bytes

    def test_pack_2bit_byte_count(self):
        """2-bit: 4 pixels per byte, so 256x128 -> 8192 bytes per frame."""
        node = SIDKITExport()
        frame = np.ones((128, 256), dtype=np.uint8) * 2
        packed = node._pack_2bit(frame)
        expected_bytes = 128 * (256 // 4)
        assert len(packed) == expected_bytes

    def test_pack_4bit_byte_count(self):
        """4-bit: 2 pixels per byte (SSD1363 nibble order)."""
        node = SIDKITExport()
        frame = np.ones((128, 256), dtype=np.uint8) * 8
        packed = node._pack_4bit(frame)
        expected_bytes = 128 * (256 // 2)
        assert len(packed) == expected_bytes


class TestSIDKITExportGrayscale:
    """Grayscale conversion."""

    def test_grayscale_conversion_shape(self):
        """RGB frame (H, W, 3) should reduce to (H, W)."""
        node = SIDKITExport()
        frame = np.random.rand(64, 128, 3).astype(np.float32)
        gray = node._to_grayscale(frame)
        assert gray.shape == (64, 128)

    def test_grayscale_already_2d(self):
        """2D frame should pass through unchanged."""
        node = SIDKITExport()
        frame = np.random.rand(64, 128).astype(np.float32)
        gray = node._to_grayscale(frame)
        assert gray.shape == (64, 128)
        np.testing.assert_array_equal(gray, frame)


# ===================================================================
# 2. KoshiOLEDPreview
# ===================================================================

@pytest.mark.pil
class TestKoshiOLEDPreview:
    """OLED preview output shape and value range."""

    def test_output_shape(self):
        """Output shape should be (B, screen_height*scale, screen_width*scale, 3)."""
        node = KoshiOLEDPreview()
        image = _make_image(batch=1, h=128, w=256)
        result = node.preview(
            image=image,
            screen_width=256,
            screen_height=128,
            bit_depth="4-bit (16 levels)",
            dither=False,
            show_pixel_grid=False,
            scale=2,
        )
        output = result[0]
        assert output.shape == (1, 128 * 2, 256 * 2, 3)

    def test_pixel_grid_gaps(self):
        """With show_pixel_grid=True and scale>=2, border pixels should be dark."""
        node = KoshiOLEDPreview()
        image = torch.ones(1, 32, 32, 3, dtype=torch.float32)
        result = node.preview(
            image=image,
            screen_width=32,
            screen_height=32,
            bit_depth="4-bit (16 levels)",
            dither=False,
            show_pixel_grid=True,
            scale=4,
        )
        output = result[0]
        # At scale=4, gap = max(1, 4//8) = 1
        # The last row of each 4-pixel block should be dark (gap line)
        gap_row_val = output[0, 3, 0, :].max().item()  # row 3 = last in first block
        assert gap_row_val <= 0.03, f"Gap row should be dark, got {gap_row_val}"

    def test_output_range(self):
        """All output values must be in [0, 1]."""
        node = KoshiOLEDPreview()
        image = _make_image(batch=2, h=64, w=64)
        result = node.preview(
            image=image,
            screen_width=64,
            screen_height=64,
            bit_depth="1-bit (mono)",
            dither=True,
            show_pixel_grid=False,
            scale=1,
        )
        output = result[0]
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0


# ===================================================================
# 3. KoshiPixelScaler
# ===================================================================

@pytest.mark.pil
class TestKoshiPixelScaler:
    """Pixel scaler with OLED presets."""

    def test_preset_ssd1363_dimensions(self):
        """SSD1363 256x128 preset should produce 256x128 output."""
        node = KoshiPixelScaler()
        images = _make_image(batch=1, h=512, w=512)
        result = node.scale(
            images=images,
            preset="SSD1363 256x128",
            custom_width=256,
            custom_height=128,
            method="lanczos",
            maintain_aspect=False,
            fill_color="black",
        )
        output = result[0]
        assert output.shape[1] == 128  # height
        assert output.shape[2] == 256  # width

    def test_maintain_aspect_ratio(self):
        """With maintain_aspect=True, image should fit inside target with letterboxing."""
        node = KoshiPixelScaler()
        images = _make_image(batch=1, h=64, w=128)  # 2:1 aspect
        result = node.scale(
            images=images,
            preset="SSD1363 256x128",
            custom_width=256,
            custom_height=128,
            method="nearest",
            maintain_aspect=True,
            fill_color="black",
        )
        output = result[0]
        # Output should be exactly the target size (padding fills the rest)
        assert output.shape[1] == 128
        assert output.shape[2] == 256


# ===================================================================
# 4. KoshiSpriteSheet
# ===================================================================

class TestKoshiSpriteSheet:
    """Sprite sheet grid layout."""

    def test_grid_layout_dimensions(self):
        """Grid layout should produce correct sheet size."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=4, h=32, w=32)
        result = node.create_sheet(
            images=images,
            layout="grid",
            max_cols=2,
            max_rows=4,
            padding=0,
            background="black",
        )
        sheet = result[0]
        cols = result[1]
        rows = result[2]
        assert cols == 2
        assert rows == 2
        assert sheet.shape[1] == 2 * 32  # height = rows * frame_h
        assert sheet.shape[2] == 2 * 32  # width = cols * frame_w

    def test_horizontal_layout(self):
        """Horizontal layout: all frames in one row."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=5, h=16, w=16)
        result = node.create_sheet(
            images=images,
            layout="horizontal",
            max_cols=32,
            max_rows=8,
            padding=0,
            background="black",
        )
        cols = result[1]
        rows = result[2]
        assert cols == 5
        assert rows == 1

    def test_frame_count_matches_batch(self):
        """Returned frame_count should equal batch size."""
        node = KoshiSpriteSheet()
        images = _make_image(batch=7, h=16, w=16)
        result = node.create_sheet(
            images=images,
            layout="auto",
            max_cols=8,
            max_rows=8,
            padding=0,
            background="black",
        )
        frame_count = result[3]
        assert frame_count == 7


# ===================================================================
# 5. KoshiXBMExport
# ===================================================================

@pytest.mark.pil
class TestKoshiXBMExport:
    """XBM export for embedded displays."""

    def test_creates_xbm_with_proper_header(self, tmp_path):
        """XBM file should contain #define _width, _height, and static unsigned char."""
        node = KoshiXBMExport()
        images = _make_image(batch=1, h=64, w=128)
        result = node.export(
            images=images,
            filename="test_sprite",
            width=128,
            height=64,
            threshold=0.5,
            invert=False,
            output_path=str(tmp_path),
        )
        file_path = result[0]
        assert os.path.isfile(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        assert "#define test_sprite_width 128" in content
        assert "#define test_sprite_height 64" in content
        assert "static unsigned char test_sprite_bits[]" in content
