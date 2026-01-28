"""Tests for nodes.flux_motion.core.easing -- cubic bezier easing presets."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest

from nodes.flux_motion.core.easing import (
    EASING_PRESETS,
    apply_easing,
    apply_easing_to_range,
    bezier_easing,
    cubic_bezier_point,
    list_easings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_PRESET_NAMES = list(EASING_PRESETS.keys())

# Presets whose control points intentionally push y outside [0, 1].
# These have y1 < 0 or y2 > 1 (back / overshoot / anticipate / bounce).
OVERSHOOT_PRESETS = {
    name
    for name, (_, y1, _, y2) in EASING_PRESETS.items()
    if y1 < 0 or y1 > 1 or y2 < 0 or y2 > 1
}

# Everything else should be monotonically non-decreasing.
MONOTONIC_PRESETS = [n for n in ALL_PRESET_NAMES if n not in OVERSHOOT_PRESETS]


# ===================================================================
# 1. Boundary conditions: f(0) == 0, f(1) == 1 for every preset
# ===================================================================

class TestBoundaryConditions:
    """Every bezier easing must start at 0 and end at 1."""

    @pytest.mark.parametrize("preset", ALL_PRESET_NAMES)
    def test_easing_at_zero(self, preset):
        result = apply_easing(0.0, preset)
        assert result == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("preset", ALL_PRESET_NAMES)
    def test_easing_at_one(self, preset):
        result = apply_easing(1.0, preset)
        assert result == pytest.approx(1.0, abs=1e-9)


# ===================================================================
# 2. Monotonicity for standard (non-overshoot) presets
# ===================================================================

class TestMonotonicity:
    """Standard presets must be non-decreasing across [0, 1]."""

    @pytest.mark.parametrize("preset", MONOTONIC_PRESETS)
    def test_monotonic_non_decreasing(self, preset):
        steps = 50
        values = [apply_easing(i / steps, preset) for i in range(steps + 1)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-6, (
                f"{preset}: value decreased at step {i}: "
                f"{values[i - 1]:.6f} -> {values[i]:.6f}"
            )


# ===================================================================
# 3. Back / overshoot / anticipate presets exceed [0, 1]
# ===================================================================

class TestOvershootPresets:
    """Presets with out-of-range control points should produce y outside [0,1]."""

    @pytest.mark.parametrize("preset", sorted(OVERSHOOT_PRESETS))
    def test_exceeds_unit_range(self, preset):
        """At least one sample in (0,1) must fall outside [0, 1]."""
        steps = 200
        values = [apply_easing(i / steps, preset) for i in range(1, steps)]
        has_below = any(v < -1e-6 for v in values)
        has_above = any(v > 1.0 + 1e-6 for v in values)
        assert has_below or has_above, (
            f"{preset}: expected y outside [0,1] but all values in "
            f"[{min(values):.4f}, {max(values):.4f}]"
        )


# ===================================================================
# 4. Unknown preset falls back to linear (identity)
# ===================================================================

class TestUnknownPreset:
    """Unknown easing name should return t unchanged."""

    def test_unknown_returns_identity_mid(self):
        assert apply_easing(0.5, "nonexistent_easing") == pytest.approx(0.5)

    def test_unknown_returns_identity_zero(self):
        assert apply_easing(0.0, "nonexistent_easing") == pytest.approx(0.0)

    def test_unknown_returns_identity_one(self):
        assert apply_easing(1.0, "nonexistent_easing") == pytest.approx(1.0)

    def test_unknown_returns_identity_arbitrary(self):
        assert apply_easing(0.73, "bogus") == pytest.approx(0.73)


# ===================================================================
# 5. apply_easing_to_range
# ===================================================================

class TestApplyEasingToRange:
    """Interpolation between two values using eased t."""

    def test_linear_range_midpoint(self):
        result = apply_easing_to_range(0.5, 10.0, 20.0, "linear")
        assert result == pytest.approx(10.0 + 10.0 * 0.5, abs=0.05)

    def test_range_at_zero(self):
        result = apply_easing_to_range(0.0, 5.0, 100.0, "easeInOut")
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_range_at_one(self):
        result = apply_easing_to_range(1.0, 5.0, 100.0, "easeInOut")
        assert result == pytest.approx(100.0, abs=1e-9)

    def test_range_negative_values(self):
        result = apply_easing_to_range(0.0, -10.0, 10.0, "linear")
        assert result == pytest.approx(-10.0, abs=1e-9)

    def test_range_reverse_direction(self):
        """from_val > to_val should interpolate downward."""
        result = apply_easing_to_range(1.0, 100.0, 0.0, "linear")
        assert result == pytest.approx(0.0, abs=1e-9)


# ===================================================================
# 6. list_easings returns all presets
# ===================================================================

class TestListEasings:
    """list_easings must expose every preset key."""

    def test_returns_all_keys(self):
        names = list_easings()
        assert set(names) == set(EASING_PRESETS.keys())

    def test_count_at_least_28(self):
        assert len(list_easings()) >= 28

    def test_returns_sorted(self):
        names = list_easings()
        assert names == sorted(names)


# ===================================================================
# 7. bezier_easing edge cases
# ===================================================================

class TestBezierEasingEdgeCases:
    """Direct tests on the bezier_easing function."""

    def test_t_zero_clamp(self):
        assert bezier_easing(0.25, 0.1, 0.25, 1.0, 0.0) == pytest.approx(0.0)

    def test_t_one_clamp(self):
        assert bezier_easing(0.25, 0.1, 0.25, 1.0, 1.0) == pytest.approx(1.0)

    def test_t_negative_clamp(self):
        assert bezier_easing(0.42, 0.0, 0.58, 1.0, -0.5) == pytest.approx(0.0)

    def test_t_above_one_clamp(self):
        assert bezier_easing(0.42, 0.0, 0.58, 1.0, 1.5) == pytest.approx(1.0)

    def test_midpoint_linear(self):
        result = bezier_easing(0.0, 0.0, 1.0, 1.0, 0.5)
        assert result == pytest.approx(0.5, abs=0.02)


# ===================================================================
# 8. cubic_bezier_point basic math
# ===================================================================

class TestCubicBezierPoint:
    """Verify the bernstein polynomial evaluation."""

    def test_at_t_zero(self):
        """B(0) should equal p0."""
        assert cubic_bezier_point(0.0, 0.0, 0.3, 0.7, 1.0) == pytest.approx(0.0)

    def test_at_t_one(self):
        """B(1) should equal p3."""
        assert cubic_bezier_point(1.0, 0.0, 0.3, 0.7, 1.0) == pytest.approx(1.0)

    def test_linear_at_midpoint(self):
        """With evenly spaced control points [0, 1/3, 2/3, 1], B(0.5) = 0.5."""
        result = cubic_bezier_point(0.5, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_constant_curve(self):
        """All control points equal -> constant value."""
        assert cubic_bezier_point(0.3, 5.0, 5.0, 5.0, 5.0) == pytest.approx(5.0)

    def test_known_value(self):
        """B(0.5) with p=[0, 0, 1, 1] = 0.5 by symmetry."""
        result = cubic_bezier_point(0.5, 0.0, 0.0, 1.0, 1.0)
        assert result == pytest.approx(0.5, abs=1e-9)


# ===================================================================
# 9. Linear preset is identity: f(t) ~ t for all t
# ===================================================================

class TestLinearIdentity:
    """The 'linear' preset should act as identity mapping."""

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_linear_is_identity(self, t):
        result = apply_easing(t, "linear")
        assert result == pytest.approx(t, abs=0.01)


# ===================================================================
# 10. Additional integration / sanity checks
# ===================================================================

class TestEasingShape:
    """Validate the qualitative shape of selected easings."""

    def test_ease_in_starts_slow(self):
        """easeIn should be below linear at t=0.25."""
        linear_val = apply_easing(0.25, "linear")
        ease_in_val = apply_easing(0.25, "easeIn")
        assert ease_in_val < linear_val

    def test_ease_out_starts_fast(self):
        """easeOut should be above linear at t=0.25."""
        linear_val = apply_easing(0.25, "linear")
        ease_out_val = apply_easing(0.25, "easeOut")
        assert ease_out_val > linear_val

    def test_ease_in_out_symmetric(self):
        """easeInOut at t=0.5 should be approximately 0.5."""
        result = apply_easing(0.5, "easeInOut")
        assert result == pytest.approx(0.5, abs=0.1)

    def test_snap_reaches_high_early(self):
        """snap preset should reach near-1 very quickly."""
        result = apply_easing(0.3, "snap")
        assert result > 0.7, f"snap at t=0.3 should be high, got {result:.4f}"
