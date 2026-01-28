"""Tests for nodes.flux_motion.semantic_motion -- KoshiSemanticMotion."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest

from nodes.flux_motion.semantic_motion import KoshiSemanticMotion, MOTION_PRESETS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    """KoshiSemanticMotion instance."""
    return KoshiSemanticMotion()


# ===================================================================
# 1. "zoom in" preset: frames have zoom > 1.0
# ===================================================================

class TestZoomInPreset:
    """'zoom in' preset produces increasing zoom values above 1.0."""

    def test_zoom_in_last_frame_above_one(self, node):
        result = node.generate(
            motion_prompt="zoom in", frames=60, preset="zoom in",
        )
        frames = result[0]["motion_frames"]
        # Last frame should have zoom > 1.0 (preset zoom_end=1.1)
        assert frames[-1].zoom > 1.0


# ===================================================================
# 2. "pan left" preset: negative translation_x
# ===================================================================

class TestPanLeftPreset:
    """'pan left' preset produces negative translation_x at last frame."""

    def test_pan_left_negative_tx(self, node):
        result = node.generate(
            motion_prompt="pan left", frames=60, preset="pan left",
        )
        frames = result[0]["motion_frames"]
        assert frames[-1].translation_x < 0.0


# ===================================================================
# 3. "static" preset: all identity values
# ===================================================================

class TestStaticPreset:
    """'static' preset produces identity motion for all frames."""

    def test_static_identity(self, node):
        result = node.generate(
            motion_prompt="static", frames=30, preset="static",
        )
        frames = result[0]["motion_frames"]
        for frame in frames:
            assert abs(frame.zoom - 1.0) < 1e-4
            assert abs(frame.angle) < 1e-4
            assert abs(frame.translation_x) < 1e-4
            assert abs(frame.translation_y) < 1e-4


# ===================================================================
# 4. Combined prompt: zoom + pan
# ===================================================================

class TestCombinedPrompt:
    """Combined 'zoom in, pan left' applies both motions."""

    def test_combined_zoom_and_pan(self, node):
        result = node.generate(
            motion_prompt="zoom in, pan left", frames=60,
        )
        frames = result[0]["motion_frames"]
        last = frames[-1]
        # Both zoom and translation should be affected
        assert last.zoom > 1.0
        assert last.translation_x < 0.0


# ===================================================================
# 5. Intensity multiplier
# ===================================================================

class TestIntensityMultiplier:
    """Intensity=2.0 amplifies motion delta compared to intensity=1.0."""

    def test_intensity_amplifies_zoom_delta(self, node):
        result_1x = node.generate(
            motion_prompt="zoom in", frames=60,
            preset="zoom in", intensity=1.0,
        )
        result_2x = node.generate(
            motion_prompt="zoom in", frames=60,
            preset="zoom in", intensity=2.0,
        )
        zoom_1x = result_1x[0]["motion_frames"][-1].zoom
        zoom_2x = result_2x[0]["motion_frames"][-1].zoom
        # Delta from 1.0 should be roughly doubled
        delta_1x = zoom_1x - 1.0
        delta_2x = zoom_2x - 1.0
        assert abs(delta_2x - 2.0 * delta_1x) < 1e-4


# ===================================================================
# 6. Correct number of frames
# ===================================================================

class TestFrameCount:
    """Returns the requested number of frames."""

    def test_frame_count(self, node):
        for n in [1, 30, 120]:
            result = node.generate(
                motion_prompt="zoom in", frames=n, preset="zoom in",
            )
            assert len(result[0]["motion_frames"]) == n


# ===================================================================
# 7. All selected presets don't crash
# ===================================================================

class TestPresetsNoCrash:
    """Selected presets execute without error."""

    SAMPLE_PRESETS = [
        "zoom in", "zoom out", "pan left", "pan right",
        "rotate left", "rotate right", "static", "push in",
        "pull out", "orbit left", "spin", "slow zoom in",
    ]

    @pytest.mark.parametrize("preset_name", SAMPLE_PRESETS)
    def test_preset_no_crash(self, node, preset_name):
        result = node.generate(
            motion_prompt=preset_name, frames=30, preset=preset_name,
        )
        assert "motion_frames" in result[0]
        assert len(result[0]["motion_frames"]) == 30


# ===================================================================
# 8. Unknown prompt produces static motion
# ===================================================================

class TestUnknownPrompt:
    """Unrecognised prompt falls back to static-like motion."""

    def test_unknown_prompt_static(self, node):
        result = node.generate(
            motion_prompt="xyzzy_nonexistent_motion", frames=30,
        )
        frames = result[0]["motion_frames"]
        # Fallback sets zoom_start=1.0, zoom_end=1.0 -> constant zoom=1.0
        for frame in frames:
            assert abs(frame.zoom - 1.0) < 1e-4


# ===================================================================
# 9. Easing changes value distribution
# ===================================================================

class TestEasingEffect:
    """Non-linear easing produces different mid-frame values than linear."""

    def test_easing_changes_midpoint(self, node):
        result_linear = node.generate(
            motion_prompt="zoom in", frames=60,
            preset="zoom in", easing="linear",
        )
        result_eased = node.generate(
            motion_prompt="zoom in", frames=60,
            preset="zoom in", easing="easeInOut",
        )
        mid_linear = result_linear[0]["motion_frames"][30].zoom
        mid_eased = result_eased[0]["motion_frames"][30].zoom
        # Easing should shift the midpoint value
        assert abs(mid_linear - mid_eased) > 1e-6


# ===================================================================
# 10. Motion frames have correct attributes
# ===================================================================

class TestMotionFrameAttributes:
    """Generated frames have all required attributes."""

    def test_frame_attributes(self, node):
        result = node.generate(
            motion_prompt="zoom in", frames=10, preset="zoom in",
        )
        frame = result[0]["motion_frames"][0]
        assert hasattr(frame, "zoom")
        assert hasattr(frame, "angle")
        assert hasattr(frame, "translation_x")
        assert hasattr(frame, "translation_y")
        assert hasattr(frame, "frame_index")
        assert frame.frame_index == 0
