"""Tests for nodes.flux_motion.core.schedule_parser module.

Covers:
  - _extract_keyframes: parsing keyframe strings with various formats
  - parse_schedule_string: interpolation, defaults, edge cases
  - MotionFrame: dataclass construction and to_dict()
  - parse_deforum_params: full parameter dict to MotionFrame list
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nodes.flux_motion.core.schedule_parser import (
    _extract_keyframes,
    parse_schedule_string,
    parse_deforum_params,
    MotionFrame,
    DEFAULTS,
)


# -----------------------------------------------------------------------
# _extract_keyframes
# -----------------------------------------------------------------------

class TestExtractKeyframes:
    """Tests for low-level keyframe string extraction."""

    def test_basic_two_keyframes(self):
        result = _extract_keyframes("0:(1.0), 30:(1.05)")
        assert result == {0: 1.0, 30: 1.05}

    def test_whitespace_variations(self):
        result = _extract_keyframes("0 : ( 1.0 ) , 30 : ( 1.05 )")
        assert result == {0: 1.0, 30: 1.05}

    def test_negative_values(self):
        result = _extract_keyframes("0:(-5.0), 15:(-2.5), 30:(0)")
        assert result == {0: -5.0, 15: -2.5, 30: 0.0}

    def test_positive_sign_prefix(self):
        result = _extract_keyframes("0:(+3.5)")
        assert result == {0: 3.5}

    def test_single_keyframe(self):
        result = _extract_keyframes("0:(1.0)")
        assert result == {0: 1.0}

    def test_empty_string(self):
        result = _extract_keyframes("")
        assert result == {}

    def test_no_parentheses(self):
        """Parser supports optional parentheses."""
        result = _extract_keyframes("0:1.0, 30:2.0")
        assert result == {0: 1.0, 30: 2.0}

    def test_integer_values(self):
        result = _extract_keyframes("0:(5), 60:(10)")
        assert result == {0: 5.0, 60: 10.0}

    def test_large_frame_numbers(self):
        result = _extract_keyframes("0:(0.0), 999:(1.0)")
        assert result == {0: 0.0, 999: 1.0}

    def test_garbage_input_returns_empty(self):
        result = _extract_keyframes("hello world no frames here")
        assert result == {}


# -----------------------------------------------------------------------
# parse_schedule_string
# -----------------------------------------------------------------------

class TestParseScheduleString:
    """Tests for schedule string to value-list conversion."""

    def test_returns_correct_length(self):
        values = parse_schedule_string("0:(1.0), 30:(2.0)", num_frames=60)
        assert len(values) == 60

    def test_empty_string_returns_default_fill(self):
        values = parse_schedule_string("", num_frames=10, default=5.0)
        assert values == [5.0] * 10

    def test_blank_whitespace_returns_default_fill(self):
        values = parse_schedule_string("   ", num_frames=8, default=0.0)
        assert values == [0.0] * 8

    def test_none_returns_default_fill(self):
        values = parse_schedule_string(None, num_frames=5, default=1.0)
        assert values == [1.0] * 5

    def test_single_keyframe_fills_constant(self):
        values = parse_schedule_string("0:(3.0)", num_frames=10)
        for v in values:
            assert v == pytest.approx(3.0, abs=1e-6)

    def test_linear_interpolation_midpoint(self):
        """Two keyframes at 0 and 10 with linear interp: midpoint should be 0.5."""
        values = parse_schedule_string(
            "0:(0.0), 10:(1.0)", num_frames=11, interpolation="linear"
        )
        assert len(values) == 11
        assert values[0] == pytest.approx(0.0, abs=1e-4)
        assert values[5] == pytest.approx(0.5, abs=1e-4)
        assert values[10] == pytest.approx(1.0, abs=1e-4)

    def test_result_is_list_of_floats(self):
        values = parse_schedule_string("0:(1.0), 30:(2.0)", num_frames=30)
        assert isinstance(values, list)
        for v in values:
            assert isinstance(v, float)

    def test_step_interpolation_holds_value(self):
        """Step interpolation should hold the last keyframe value."""
        values = parse_schedule_string(
            "0:(0.0), 5:(10.0)", num_frames=10, interpolation="step"
        )
        # Frames 0-4 should hold 0.0, frames 5-9 should hold 10.0
        for i in range(5):
            assert values[i] == pytest.approx(0.0, abs=1e-6)
        for i in range(5, 10):
            assert values[i] == pytest.approx(10.0, abs=1e-6)

    def test_custom_default_value(self):
        values = parse_schedule_string("not_valid", num_frames=5, default=7.7)
        assert values == [7.7] * 5


# -----------------------------------------------------------------------
# MotionFrame
# -----------------------------------------------------------------------

class TestMotionFrame:
    """Tests for the MotionFrame dataclass."""

    def test_required_frame_index(self):
        frame = MotionFrame(frame_index=0)
        assert frame.frame_index == 0

    def test_default_values(self):
        frame = MotionFrame(frame_index=0)
        assert frame.zoom == 1.0
        assert frame.angle == 0.0
        assert frame.translation_x == 0.0
        assert frame.translation_y == 0.0
        assert frame.translation_z == 0.0
        assert frame.strength == 0.65
        assert frame.prompt is None

    def test_custom_values(self):
        frame = MotionFrame(
            frame_index=10,
            zoom=1.5,
            angle=45.0,
            translation_x=10.0,
            translation_y=-5.0,
            translation_z=2.0,
            strength=0.8,
            prompt="a cat",
        )
        assert frame.frame_index == 10
        assert frame.zoom == 1.5
        assert frame.angle == 45.0
        assert frame.translation_x == 10.0
        assert frame.translation_y == -5.0
        assert frame.translation_z == 2.0
        assert frame.strength == 0.8
        assert frame.prompt == "a cat"

    def test_to_dict_keys(self):
        frame = MotionFrame(frame_index=0, zoom=2.0, angle=90.0)
        result = frame.to_dict()
        expected_keys = {"zoom", "angle", "translation_x", "translation_y", "translation_z"}
        assert set(result.keys()) == expected_keys

    def test_to_dict_values(self):
        frame = MotionFrame(frame_index=5, zoom=1.2, angle=30.0, translation_x=4.0)
        d = frame.to_dict()
        assert d["zoom"] == 1.2
        assert d["angle"] == 30.0
        assert d["translation_x"] == 4.0
        assert d["translation_y"] == 0.0
        assert d["translation_z"] == 0.0


# -----------------------------------------------------------------------
# parse_deforum_params
# -----------------------------------------------------------------------

class TestParseDeforumParams:
    """Tests for full Deforum parameter dict parsing."""

    def test_returns_correct_frame_count(self):
        params = {"zoom": "0:(1.0)"}
        frames = parse_deforum_params(params, num_frames=30)
        assert len(frames) == 30

    def test_all_elements_are_motion_frames(self):
        params = {"zoom": "0:(1.0)"}
        frames = parse_deforum_params(params, num_frames=10)
        for f in frames:
            assert isinstance(f, MotionFrame)

    def test_frame_indices_sequential(self):
        params = {}
        frames = parse_deforum_params(params, num_frames=5)
        for i, f in enumerate(frames):
            assert f.frame_index == i

    def test_defaults_when_no_params(self):
        frames = parse_deforum_params({}, num_frames=3)
        for f in frames:
            assert f.zoom == DEFAULTS["zoom"]
            assert f.angle == DEFAULTS["angle"]
            assert f.translation_x == DEFAULTS["translation_x"]
            assert f.translation_y == DEFAULTS["translation_y"]
            assert f.translation_z == DEFAULTS["translation_z"]
            assert f.strength == DEFAULTS["strength"]

    def test_partial_params_fills_defaults(self):
        params = {"zoom": "0:(2.0)"}
        frames = parse_deforum_params(params, num_frames=5)
        for f in frames:
            assert f.zoom == pytest.approx(2.0, abs=1e-6)
            assert f.angle == DEFAULTS["angle"]
            assert f.strength == DEFAULTS["strength"]

    def test_numeric_param_constant_fill(self):
        """Passing a raw number fills every frame with that constant."""
        params = {"zoom": 1.5, "angle": 10.0}
        frames = parse_deforum_params(params, num_frames=4)
        for f in frames:
            assert f.zoom == pytest.approx(1.5, abs=1e-6)
            assert f.angle == pytest.approx(10.0, abs=1e-6)

    def test_prompt_expansion(self):
        params = {"prompts": {0: "forest", 5: "ocean"}}
        frames = parse_deforum_params(params, num_frames=10)
        # Frames 0-4 should get "forest", frames 5-9 should get "ocean"
        for i in range(5):
            assert frames[i].prompt == "forest"
        for i in range(5, 10):
            assert frames[i].prompt == "ocean"

    def test_empty_prompts_yields_none(self):
        frames = parse_deforum_params({}, num_frames=3)
        for f in frames:
            assert f.prompt is None


# -----------------------------------------------------------------------
# DEFAULTS constant
# -----------------------------------------------------------------------

class TestDefaults:
    """Verify the module-level DEFAULTS dict."""

    def test_defaults_has_required_keys(self):
        required = {"zoom", "angle", "translation_x", "translation_y", "translation_z", "strength"}
        assert required.issubset(set(DEFAULTS.keys()))

    def test_defaults_values_are_numeric(self):
        for key, val in DEFAULTS.items():
            assert isinstance(val, (int, float)), f"DEFAULTS[{key!r}] is not numeric"
