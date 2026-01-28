"""Cross-verify Koshi easing functions against Deforum2026 easing.

Both repos implement CSS-style cubic bezier easing with identical preset
dictionaries. These tests confirm output parity so changes in either repo
are caught immediately.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Koshi easing -- always available
from nodes.flux_motion.core.easing import (
    apply_easing as koshi_easing,
    EASING_PRESETS as koshi_presets,
    bezier_easing as koshi_bezier,
    list_easings as koshi_list_easings,
)

# t-values to test (boundaries + interior)
T_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Presets expected in both repositories
SHARED_PRESETS = [
    "linear", "ease", "easeIn", "easeOut", "easeInOut",
    "easeInSine", "easeOutSine", "easeInOutSine",
    "easeInQuad", "easeOutQuad", "easeInOutQuad",
    "easeInCubic", "easeOutCubic", "easeInOutCubic",
]

# All presets in Koshi (superset used for standalone boundary tests)
ALL_KOSHI_PRESETS = list(koshi_presets.keys())


def _try_import_deforum_easing():
    """Attempt to import Deforum2026 easing, return module or None."""
    try:
        from core.easing import (
            apply_easing as deforum_easing,
            EASING_PRESETS as deforum_presets,
        )
        return deforum_easing, deforum_presets
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# 1. Parametrized parity: shared presets x t-values
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("preset", SHARED_PRESETS)
@pytest.mark.parametrize("t", T_VALUES)
def test_easing_parity_shared_presets(preset, t):
    """Koshi and Deforum must produce identical eased values."""
    deforum_easing, _ = _try_import_deforum_easing()
    if deforum_easing is None:
        pytest.skip("Deforum2026 not available")

    koshi_val = koshi_easing(t, preset)
    deforum_val = deforum_easing(t, preset)
    assert abs(koshi_val - deforum_val) < 1e-6, (
        f"preset={preset}, t={t}: koshi={koshi_val}, deforum={deforum_val}"
    )


# ---------------------------------------------------------------------------
# 2. Both have same preset names
# ---------------------------------------------------------------------------

@pytest.mark.deforum
def test_preset_keys_match():
    """Koshi and Deforum EASING_PRESETS dicts must have identical keys."""
    _, deforum_presets = _try_import_deforum_easing()
    if deforum_presets is None:
        pytest.skip("Deforum2026 not available")

    koshi_keys = set(koshi_presets.keys())
    deforum_keys = set(deforum_presets.keys())
    assert koshi_keys == deforum_keys, (
        f"Only in Koshi: {koshi_keys - deforum_keys}, "
        f"Only in Deforum: {deforum_keys - koshi_keys}"
    )


# ---------------------------------------------------------------------------
# 3. Boundary values: t=0 -> 0, t=1 -> ~1 for all presets
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("preset", ALL_KOSHI_PRESETS)
def test_koshi_boundary_zero(preset):
    """apply_easing(0, preset) must return 0 for all presets."""
    assert koshi_easing(0.0, preset) == 0.0


@pytest.mark.deforum
@pytest.mark.parametrize("preset", ALL_KOSHI_PRESETS)
def test_koshi_boundary_one(preset):
    """apply_easing(1, preset) must return ~1 for all presets."""
    val = koshi_easing(1.0, preset)
    assert abs(val - 1.0) < 1e-6, f"preset={preset}: easing(1.0) = {val}"


@pytest.mark.deforum
@pytest.mark.parametrize("preset", SHARED_PRESETS)
def test_deforum_boundary_zero(preset):
    """Deforum apply_easing(0, preset) must return 0 for shared presets."""
    deforum_easing, _ = _try_import_deforum_easing()
    if deforum_easing is None:
        pytest.skip("Deforum2026 not available")
    assert deforum_easing(0.0, preset) == 0.0


@pytest.mark.deforum
@pytest.mark.parametrize("preset", SHARED_PRESETS)
def test_deforum_boundary_one(preset):
    """Deforum apply_easing(1, preset) must return ~1 for shared presets."""
    deforum_easing, _ = _try_import_deforum_easing()
    if deforum_easing is None:
        pytest.skip("Deforum2026 not available")
    val = deforum_easing(1.0, preset)
    assert abs(val - 1.0) < 1e-6, f"preset={preset}: easing(1.0) = {val}"


# ---------------------------------------------------------------------------
# 4. Linear preset: identity function
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("t", T_VALUES)
def test_koshi_linear_is_identity(t):
    """Linear easing must return the input value unchanged."""
    val = koshi_easing(t, "linear")
    assert abs(val - t) < 1e-6, f"linear({t}) = {val}"


@pytest.mark.deforum
@pytest.mark.parametrize("t", T_VALUES)
def test_deforum_linear_is_identity(t):
    """Deforum linear easing must return the input value unchanged."""
    deforum_easing, _ = _try_import_deforum_easing()
    if deforum_easing is None:
        pytest.skip("Deforum2026 not available")
    val = deforum_easing(t, "linear")
    assert abs(val - t) < 1e-6, f"linear({t}) = {val}"


# ---------------------------------------------------------------------------
# 5. Control point values match between repos
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("preset", SHARED_PRESETS)
def test_control_points_match(preset):
    """Bezier control points must be identical for shared presets."""
    _, deforum_presets = _try_import_deforum_easing()
    if deforum_presets is None:
        pytest.skip("Deforum2026 not available")

    koshi_cp = koshi_presets[preset]
    deforum_cp = deforum_presets[preset]
    assert koshi_cp == deforum_cp, (
        f"preset={preset}: koshi={koshi_cp}, deforum={deforum_cp}"
    )


# ---------------------------------------------------------------------------
# 6. Document Deforum extras (functions that exist in Deforum but not Koshi)
# ---------------------------------------------------------------------------

@pytest.mark.deforum
def test_deforum_has_simple_easings():
    """Deforum has SIMPLE_EASINGS dict; document presence for future parity."""
    try:
        from core.easing import SIMPLE_EASINGS
    except ImportError:
        pytest.skip("Deforum2026 not available")

    assert isinstance(SIMPLE_EASINGS, dict)
    assert len(SIMPLE_EASINGS) > 0
    expected_funcs = [
        "ease_in_quad", "ease_out_quad", "ease_in_out_quad",
        "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
        "ease_in_expo", "ease_out_expo", "ease_in_out_expo",
    ]
    for name in expected_funcs:
        assert name in SIMPLE_EASINGS, f"Missing simple easing: {name}"


@pytest.mark.deforum
def test_deforum_has_get_easing_points():
    """Deforum has get_easing_points(); Koshi does not expose this separately."""
    try:
        from core.easing import get_easing_points
    except ImportError:
        pytest.skip("Deforum2026 not available")

    points = get_easing_points("linear")
    assert points == (0.0, 0.0, 1.0, 1.0)
