"""Tests for nodes.flux_motion.schedule -- KoshiSchedule and KoshiScheduleMulti."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import numpy as np

from nodes.flux_motion.schedule import KoshiSchedule, KoshiScheduleMulti


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schedule_node():
    """KoshiSchedule instance."""
    return KoshiSchedule()


@pytest.fixture
def multi_node():
    """KoshiScheduleMulti instance."""
    return KoshiScheduleMulti()


# ===================================================================
# KoshiSchedule
# ===================================================================

class TestKoshiScheduleBasicParse:
    """Parsed schedule returns dict with expected keys."""

    def test_returns_dict_with_expected_keys(self, schedule_node):
        result = schedule_node.parse(
            schedule_string="0:(1.0), 30:(1.05), 60:(1.0)",
            max_frames=120,
            interpolation="linear",
            easing="none",
        )
        schedule = result[0]
        expected_keys = {"name", "frames", "values", "interpolation", "easing", "raw"}
        assert expected_keys == set(schedule.keys())


class TestKoshiScheduleValueCount:
    """Parsed values list length equals max_frames."""

    def test_value_count_equals_max_frames(self, schedule_node):
        max_frames = 60
        result = schedule_node.parse(
            schedule_string="0:(1.0), 30:(1.05)",
            max_frames=max_frames,
            interpolation="linear",
            easing="none",
        )
        schedule = result[0]
        assert len(schedule["values"]) == max_frames


class TestKoshiScheduleTwoKeyframes:
    """Two-keyframe interpolation: first and last values match keyframes."""

    def test_first_and_last_values_match_keyframes(self, schedule_node):
        result = schedule_node.parse(
            schedule_string="0:(2.0), 59:(5.0)",
            max_frames=60,
            interpolation="linear",
            easing="none",
        )
        values = result[0]["values"]
        assert abs(values[0] - 2.0) < 1e-4
        assert abs(values[59] - 5.0) < 1e-4


class TestKoshiScheduleLinearMiddle:
    """Linear interpolation produces correct midpoint value."""

    def test_midpoint_value_linear(self, schedule_node):
        result = schedule_node.parse(
            schedule_string="0:(0.0), 100:(100.0)",
            max_frames=101,
            interpolation="linear",
            easing="none",
        )
        values = result[0]["values"]
        assert abs(values[50] - 50.0) < 1.0


class TestKoshiScheduleSingleKeyframe:
    """Single keyframe produces constant value array."""

    def test_single_keyframe_constant(self, schedule_node):
        result = schedule_node.parse(
            schedule_string="0:(3.14)",
            max_frames=60,
            interpolation="linear",
            easing="none",
        )
        values = result[0]["values"]
        for v in values:
            assert abs(v - 3.14) < 1e-4


# ===================================================================
# KoshiScheduleMulti
# ===================================================================

class TestKoshiScheduleMultiReturnKey:
    """Returns dict with 'motion_frames' key."""

    def test_returns_motion_frames_key(self, multi_node):
        result = multi_node.parse(max_frames=30, interpolation="linear")
        schedule = result[0]
        assert "motion_frames" in schedule


class TestKoshiScheduleMultiFrameCount:
    """motion_frames length equals max_frames."""

    def test_motion_frames_length(self, multi_node):
        max_frames = 45
        result = multi_node.parse(max_frames=max_frames, interpolation="linear")
        schedule = result[0]
        assert len(schedule["motion_frames"]) == max_frames


class TestKoshiScheduleMultiZoomParsed:
    """Zoom schedule is parsed into motion frames."""

    def test_zoom_schedule_parsed(self, multi_node):
        result = multi_node.parse(
            max_frames=30,
            interpolation="linear",
            zoom="0:(1.0), 29:(2.0)",
        )
        frames = result[0]["motion_frames"]
        assert abs(frames[0].zoom - 1.0) < 1e-4
        assert abs(frames[-1].zoom - 2.0) < 1e-4


class TestKoshiScheduleMultiDefaults:
    """Default values used for unspecified schedule strings."""

    def test_default_values(self, multi_node):
        result = multi_node.parse(max_frames=10, interpolation="linear")
        frame = result[0]["motion_frames"][0]
        assert abs(frame.zoom - 1.0) < 1e-4
        assert abs(frame.angle - 0.0) < 1e-4
        assert abs(frame.translation_x - 0.0) < 1e-4
        assert abs(frame.translation_y - 0.0) < 1e-4


class TestKoshiScheduleMultiAttributes:
    """Motion frames have correct attributes."""

    def test_motion_frame_attributes(self, multi_node):
        result = multi_node.parse(max_frames=10, interpolation="linear")
        frame = result[0]["motion_frames"][0]
        assert hasattr(frame, "zoom")
        assert hasattr(frame, "angle")
        assert hasattr(frame, "translation_x")
        assert hasattr(frame, "translation_y")
        assert hasattr(frame, "frame_index")


class TestKoshiScheduleMultiInterpolation:
    """Different interpolation methods work without error."""

    @pytest.mark.parametrize("method", ["linear", "cubic", "step"])
    def test_interpolation_methods(self, multi_node, method):
        result = multi_node.parse(
            max_frames=30,
            interpolation=method,
            zoom="0:(1.0), 29:(1.5)",
        )
        assert len(result[0]["motion_frames"]) == 30


class TestKoshiScheduleMultiFrameIndex:
    """All frames have correct sequential frame_index."""

    def test_frame_indices_sequential(self, multi_node):
        result = multi_node.parse(max_frames=20, interpolation="linear")
        frames = result[0]["motion_frames"]
        for i, frame in enumerate(frames):
            assert frame.frame_index == i
