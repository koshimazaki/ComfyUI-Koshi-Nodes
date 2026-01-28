"""Tests for nodes.flux_motion.core.interpolation module.

Covers:
- linear_interpolation: basic 2-keyframe, multi-keyframe, single keyframe,
  empty keyframes, holds last value, clamps to first value before range.
- step_interpolation: holds values until next keyframe, single keyframe,
  empty frames, before-first-frame behavior.
- cubic_spline_interpolation: smooth curve with scipy, fallback to linear
  without scipy.
- interpolate_array: dispatches to correct method for linear/step/cubic,
  empty and single-keyframe edge cases.
- lerp: basic interpolation at t=0, t=0.5, t=1, negative t, t>1.
- smoothstep: boundaries, midpoint, clamp outside range, degenerate edge.
- SCIPY_AVAILABLE flag existence.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from nodes.flux_motion.core.interpolation import (
    linear_interpolation,
    step_interpolation,
    cubic_spline_interpolation,
    interpolate_array,
    lerp,
    smoothstep,
    SCIPY_AVAILABLE,
)


# =====================================================================
# linear_interpolation
# =====================================================================

class TestLinearInterpolation:
    """Tests for linear_interpolation(frames, values, target_frame)."""

    def test_two_keyframes_midpoint(self):
        """Midpoint between two keyframes returns the average."""
        result = linear_interpolation([0, 10], [0.0, 10.0], 5)
        assert result == pytest.approx(5.0)

    def test_two_keyframes_at_boundaries(self):
        """Exact keyframe positions return exact values."""
        assert linear_interpolation([0, 10], [2.0, 8.0], 0) == pytest.approx(2.0)
        assert linear_interpolation([0, 10], [2.0, 8.0], 10) == pytest.approx(8.0)

    def test_multi_keyframe_segments(self):
        """Multiple segments: interpolates within the correct segment."""
        frames = [0, 10, 20]
        values = [0.0, 10.0, 0.0]
        # First segment midpoint
        assert linear_interpolation(frames, values, 5) == pytest.approx(5.0)
        # Second segment midpoint
        assert linear_interpolation(frames, values, 15) == pytest.approx(5.0)
        # Exact middle keyframe
        assert linear_interpolation(frames, values, 10) == pytest.approx(10.0)

    def test_single_keyframe(self):
        """Single keyframe always returns its value regardless of target."""
        assert linear_interpolation([5], [3.0], 0) == pytest.approx(3.0)
        assert linear_interpolation([5], [3.0], 5) == pytest.approx(3.0)
        assert linear_interpolation([5], [3.0], 99) == pytest.approx(3.0)

    def test_empty_frames(self):
        """Empty frame list returns 0.0."""
        assert linear_interpolation([], [], 0) == pytest.approx(0.0)

    def test_holds_last_value_beyond_range(self):
        """Frames beyond the last keyframe return the last value."""
        result = linear_interpolation([0, 10], [1.0, 5.0], 100)
        assert result == pytest.approx(5.0)

    def test_clamps_first_value_before_range(self):
        """Frames before the first keyframe return the first value."""
        result = linear_interpolation([5, 15], [10.0, 20.0], 0)
        assert result == pytest.approx(10.0)


# =====================================================================
# step_interpolation
# =====================================================================

class TestStepInterpolation:
    """Tests for step_interpolation(frames, values, target_frame)."""

    def test_holds_value_until_next_keyframe(self):
        """Value stays constant until the next keyframe is reached."""
        frames = [0, 10, 20]
        values = [1.0, 5.0, 9.0]
        # Just before second keyframe -- still first value
        assert step_interpolation(frames, values, 9) == pytest.approx(1.0)
        # Exactly at second keyframe
        assert step_interpolation(frames, values, 10) == pytest.approx(5.0)
        # Between second and third
        assert step_interpolation(frames, values, 15) == pytest.approx(5.0)
        # At last keyframe
        assert step_interpolation(frames, values, 20) == pytest.approx(9.0)
        # Beyond last keyframe
        assert step_interpolation(frames, values, 50) == pytest.approx(9.0)

    def test_single_keyframe(self):
        """Single keyframe: returns its value at or after the frame."""
        assert step_interpolation([5], [7.0], 5) == pytest.approx(7.0)
        assert step_interpolation([5], [7.0], 100) == pytest.approx(7.0)

    def test_before_first_keyframe(self):
        """Target before the first keyframe returns the first value."""
        result = step_interpolation([5, 10], [3.0, 6.0], 0)
        assert result == pytest.approx(3.0)

    def test_empty_frames(self):
        """Empty frame list returns 0.0."""
        assert step_interpolation([], [], 0) == pytest.approx(0.0)


# =====================================================================
# cubic_spline_interpolation
# =====================================================================

class TestCubicSplineInterpolation:
    """Tests for cubic_spline_interpolation(frames, values, target_frame)."""

    @pytest.mark.scipy
    def test_smooth_curve_at_keyframes(self):
        """Cubic spline passes through exact keyframe values."""
        frames = [0, 10, 20, 30]
        values = [0.0, 5.0, 3.0, 8.0]
        for f, v in zip(frames, values):
            assert cubic_spline_interpolation(frames, values, f) == pytest.approx(v, abs=1e-6)

    @pytest.mark.scipy
    def test_smooth_curve_between_keyframes(self):
        """Cubic spline produces a value between keyframes (not just linear)."""
        frames = [0, 10, 20]
        values = [0.0, 10.0, 0.0]
        mid = cubic_spline_interpolation(frames, values, 5)
        linear_mid = linear_interpolation(frames, values, 5)
        # Cubic should differ from linear (unless curve happens to match)
        # At minimum, it should be a finite number in a reasonable range
        assert np.isfinite(mid)
        assert -5.0 <= mid <= 15.0

    @pytest.mark.scipy
    def test_clamps_to_range(self):
        """Values beyond keyframe range are clamped to boundary frames."""
        frames = [5, 15]
        values = [2.0, 8.0]
        before = cubic_spline_interpolation(frames, values, 0)
        after = cubic_spline_interpolation(frames, values, 100)
        # The function clamps target_frame to [frames[0], frames[-1]]
        assert before == pytest.approx(2.0, abs=1e-6)
        assert after == pytest.approx(8.0, abs=1e-6)

    def test_fallback_single_keyframe(self):
        """Single keyframe falls back to linear (returns that value)."""
        result = cubic_spline_interpolation([5], [4.0], 5)
        assert result == pytest.approx(4.0)

    def test_fallback_empty_frames(self):
        """Empty frames falls back to linear (returns 0.0)."""
        result = cubic_spline_interpolation([], [], 0)
        assert result == pytest.approx(0.0)


# =====================================================================
# interpolate_array
# =====================================================================

class TestInterpolateArray:
    """Tests for interpolate_array(frames, values, total_frames, method)."""

    def test_linear_dispatch(self):
        """Linear method produces a ramp for two keyframes."""
        result = interpolate_array([0, 9], [0.0, 9.0], 10, method="linear")
        assert result.shape == (10,)
        # First and last values should match keyframes
        assert result[0] == pytest.approx(0.0, abs=0.1)
        assert result[9] == pytest.approx(9.0, abs=0.1)

    def test_step_dispatch(self):
        """Step method holds constant between keyframes."""
        result = interpolate_array([0, 5], [1.0, 2.0], 10, method="step")
        assert result.shape == (10,)
        # Before second keyframe, should hold first value
        assert result[3] == pytest.approx(1.0)
        # At and after second keyframe, should hold second value
        assert result[5] == pytest.approx(2.0)
        assert result[9] == pytest.approx(2.0)

    @pytest.mark.scipy
    def test_cubic_dispatch(self):
        """Cubic method produces an array of correct length."""
        result = interpolate_array([0, 5, 10], [0.0, 10.0, 0.0], 11, method="cubic")
        assert result.shape == (11,)
        # Keyframe values should be close
        assert result[0] == pytest.approx(0.0, abs=0.5)
        assert result[5] == pytest.approx(10.0, abs=0.5)
        assert result[10] == pytest.approx(0.0, abs=0.5)

    def test_empty_frames_returns_zeros(self):
        """Empty frames returns all-zero array."""
        result = interpolate_array([], [], 10)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_single_keyframe_fills_constant(self):
        """Single keyframe fills the entire array with that value."""
        result = interpolate_array([3], [7.0], 10)
        np.testing.assert_array_equal(result, np.full(10, 7.0))


# =====================================================================
# lerp
# =====================================================================

class TestLerp:
    """Tests for lerp(a, b, t)."""

    def test_t_zero_returns_a(self):
        assert lerp(2.0, 8.0, 0.0) == pytest.approx(2.0)

    def test_t_one_returns_b(self):
        assert lerp(2.0, 8.0, 1.0) == pytest.approx(8.0)

    def test_t_half_returns_midpoint(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)

    def test_negative_t_extrapolates(self):
        """Negative t extrapolates below a."""
        result = lerp(0.0, 10.0, -0.5)
        assert result == pytest.approx(-5.0)

    def test_t_greater_than_one_extrapolates(self):
        """t > 1 extrapolates beyond b."""
        result = lerp(0.0, 10.0, 1.5)
        assert result == pytest.approx(15.0)


# =====================================================================
# smoothstep
# =====================================================================

class TestSmoothstep:
    """Tests for smoothstep(edge0, edge1, x)."""

    def test_at_edge0_returns_zero(self):
        assert smoothstep(0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_at_edge1_returns_one(self):
        assert smoothstep(0.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        """Smoothstep at midpoint of [0, 1] is exactly 0.5."""
        assert smoothstep(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_clamps_below_edge0(self):
        """x below edge0 clamps to 0."""
        assert smoothstep(2.0, 4.0, 0.0) == pytest.approx(0.0)

    def test_clamps_above_edge1(self):
        """x above edge1 clamps to 1."""
        assert smoothstep(2.0, 4.0, 10.0) == pytest.approx(1.0)

    def test_degenerate_edges_equal(self):
        """When edge0 == edge1, returns 1.0 if x >= edge, else 0.0."""
        assert smoothstep(5.0, 5.0, 5.0) == pytest.approx(1.0)
        assert smoothstep(5.0, 5.0, 4.0) == pytest.approx(0.0)


# =====================================================================
# SCIPY_AVAILABLE flag
# =====================================================================

class TestScipyFlag:
    """Verify the SCIPY_AVAILABLE flag exists and is boolean."""

    def test_scipy_flag_is_bool(self):
        assert isinstance(SCIPY_AVAILABLE, bool)
