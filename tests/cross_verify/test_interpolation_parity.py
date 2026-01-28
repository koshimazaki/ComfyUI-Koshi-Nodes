"""Cross-verify Koshi interpolation functions against Deforum2026 interpolation.

Both repos implement linear_interpolation, step_interpolation, lerp, and
smoothstep with identical signatures. These tests confirm numerical parity.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Koshi interpolation -- always available
from nodes.flux_motion.core.interpolation import (
    linear_interpolation as koshi_linear,
    step_interpolation as koshi_step,
    lerp as koshi_lerp,
    smoothstep as koshi_smoothstep,
)


def _try_import_deforum_interpolation():
    """Attempt to import Deforum2026 interpolation, return dict or None."""
    try:
        from core.interpolation import (
            linear_interpolation as deforum_linear,
            step_interpolation as deforum_step,
            lerp as deforum_lerp,
            smoothstep as deforum_smoothstep,
        )
        return {
            "linear_interpolation": deforum_linear,
            "step_interpolation": deforum_step,
            "lerp": deforum_lerp,
            "smoothstep": deforum_smoothstep,
        }
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

KEYFRAMES = [0, 10, 20, 30]
KEYVALUES = [0.0, 5.0, 2.0, 8.0]
TARGET_FRAMES = [0, 5, 10, 15, 20, 25, 30, 35]


# ---------------------------------------------------------------------------
# 1. linear_interpolation parity
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("target", TARGET_FRAMES)
def test_linear_interpolation_parity(target):
    """Koshi and Deforum linear_interpolation produce identical output."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    koshi_val = koshi_linear(KEYFRAMES, KEYVALUES, target)
    deforum_val = df["linear_interpolation"](KEYFRAMES, KEYVALUES, target)
    assert abs(koshi_val - deforum_val) < 1e-10, (
        f"frame={target}: koshi={koshi_val}, deforum={deforum_val}"
    )


# ---------------------------------------------------------------------------
# 2. step_interpolation parity
# ---------------------------------------------------------------------------

@pytest.mark.deforum
@pytest.mark.parametrize("target", TARGET_FRAMES)
def test_step_interpolation_parity(target):
    """Koshi and Deforum step_interpolation produce identical output."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    koshi_val = koshi_step(KEYFRAMES, KEYVALUES, target)
    deforum_val = df["step_interpolation"](KEYFRAMES, KEYVALUES, target)
    assert abs(koshi_val - deforum_val) < 1e-10, (
        f"frame={target}: koshi={koshi_val}, deforum={deforum_val}"
    )


# ---------------------------------------------------------------------------
# 3. lerp parity
# ---------------------------------------------------------------------------

LERP_CASES = [
    (0.0, 10.0, 0.0),
    (0.0, 10.0, 0.5),
    (0.0, 10.0, 1.0),
    (-5.0, 5.0, 0.25),
    (100.0, 200.0, 0.75),
]


@pytest.mark.deforum
@pytest.mark.parametrize("a,b,t", LERP_CASES)
def test_lerp_parity(a, b, t):
    """Koshi and Deforum lerp produce identical output."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    koshi_val = koshi_lerp(a, b, t)
    deforum_val = df["lerp"](a, b, t)
    assert abs(koshi_val - deforum_val) < 1e-10, (
        f"lerp({a}, {b}, {t}): koshi={koshi_val}, deforum={deforum_val}"
    )


# ---------------------------------------------------------------------------
# 4. smoothstep parity
# ---------------------------------------------------------------------------

SMOOTHSTEP_CASES = [
    (0.0, 1.0, -0.5),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.25),
    (0.0, 1.0, 0.5),
    (0.0, 1.0, 0.75),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 1.5),
    (2.0, 8.0, 5.0),
]


@pytest.mark.deforum
@pytest.mark.parametrize("edge0,edge1,x", SMOOTHSTEP_CASES)
def test_smoothstep_parity(edge0, edge1, x):
    """Koshi and Deforum smoothstep produce identical output."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    koshi_val = koshi_smoothstep(edge0, edge1, x)
    deforum_val = df["smoothstep"](edge0, edge1, x)
    assert abs(koshi_val - deforum_val) < 1e-10, (
        f"smoothstep({edge0}, {edge1}, {x}): koshi={koshi_val}, deforum={deforum_val}"
    )


# ---------------------------------------------------------------------------
# 5. Document missing functions: Koshi lacks inverse_lerp, remap, smootherstep
# ---------------------------------------------------------------------------

@pytest.mark.deforum
def test_deforum_has_inverse_lerp():
    """Deforum has inverse_lerp; Koshi interpolation module does not."""
    try:
        from core.interpolation import inverse_lerp
    except ImportError:
        pytest.skip("Deforum2026 not available")

    # Verify it works correctly
    result = inverse_lerp(0.0, 10.0, 5.0)
    assert abs(result - 0.5) < 1e-10

    # Confirm Koshi does NOT have it
    try:
        from nodes.flux_motion.core.interpolation import inverse_lerp as _koshi_inv
        has_it = True
    except ImportError:
        has_it = False
    assert not has_it, "Koshi now has inverse_lerp -- update parity tests"


@pytest.mark.deforum
def test_deforum_has_remap():
    """Deforum has remap; Koshi interpolation module does not."""
    try:
        from core.interpolation import remap
    except ImportError:
        pytest.skip("Deforum2026 not available")

    result = remap(5.0, 0.0, 10.0, 100.0, 200.0)
    assert abs(result - 150.0) < 1e-10

    try:
        from nodes.flux_motion.core.interpolation import remap as _koshi_remap
        has_it = True
    except ImportError:
        has_it = False
    assert not has_it, "Koshi now has remap -- update parity tests"


@pytest.mark.deforum
def test_deforum_has_smootherstep():
    """Deforum has smootherstep (Perlin); Koshi does not."""
    try:
        from core.interpolation import smootherstep
    except ImportError:
        pytest.skip("Deforum2026 not available")

    result = smootherstep(0.0, 1.0, 0.5)
    assert abs(result - 0.5) < 1e-6  # smootherstep(0.5) = 0.5

    try:
        from nodes.flux_motion.core.interpolation import smootherstep as _koshi_ss
        has_it = True
    except ImportError:
        has_it = False
    assert not has_it, "Koshi now has smootherstep -- update parity tests"


# ---------------------------------------------------------------------------
# 6. Edge case: empty / single keyframe
# ---------------------------------------------------------------------------

@pytest.mark.deforum
def test_empty_keyframes_parity():
    """Both repos return 0.0 for empty keyframe list."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    assert koshi_linear([], [], 5) == df["linear_interpolation"]([], [], 5) == 0.0
    assert koshi_step([], [], 5) == df["step_interpolation"]([], [], 5) == 0.0


@pytest.mark.deforum
def test_single_keyframe_parity():
    """Both repos return the sole value for a single keyframe."""
    df = _try_import_deforum_interpolation()
    if df is None:
        pytest.skip("Deforum2026 not available")

    assert koshi_linear([10], [7.5], 0) == df["linear_interpolation"]([10], [7.5], 0) == 7.5
    assert koshi_linear([10], [7.5], 20) == df["linear_interpolation"]([10], [7.5], 20) == 7.5
