"""Tests for export nodes: KoshiSpriteSheet, KoshiOLEDScreen."""

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

from nodes.export.oled_screen import KoshiSpriteSheet, KoshiOLEDScreen


def _make_image(batch=1, h=64, w=64, c=3):
    """Create a ComfyUI-format image tensor [B, H, W, C] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(batch, h, w, c, dtype=torch.float32)


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
        assert sheet.shape[1] == 2 * 32
        assert sheet.shape[2] == 2 * 32

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


class TestKoshiOLEDScreen:
    """OLED screen node interface."""

    def test_has_comfyui_interface(self):
        assert hasattr(KoshiOLEDScreen, "INPUT_TYPES")
        assert hasattr(KoshiOLEDScreen, "RETURN_TYPES")
        assert hasattr(KoshiOLEDScreen, "FUNCTION")

    def test_input_types_valid(self):
        inputs = KoshiOLEDScreen.INPUT_TYPES()
        assert "required" in inputs
        assert "images" in inputs["required"]
