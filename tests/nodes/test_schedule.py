"""Tests for nodes.flux_motion.schedule -- KoshiSchedule."""

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

from nodes.flux_motion.schedule import KoshiSchedule


@pytest.fixture
def schedule_node():
    """KoshiSchedule instance."""
    return KoshiSchedule()


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


