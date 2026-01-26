"""Bezier easing functions with 30+ presets."""

from typing import Tuple, Dict, List


def cubic_bezier_point(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Calculate point on cubic bezier curve at parameter t."""
    mt = 1 - t
    return mt*mt*mt*p0 + 3*mt*mt*t*p1 + 3*mt*t*t*p2 + t*t*t*p3


def bezier_easing(x1: float, y1: float, x2: float, y2: float, t: float) -> float:
    """CSS-style cubic bezier easing."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        x = cubic_bezier_point(mid, 0, x1, x2, 1)
        if x < t:
            low = mid
        else:
            high = mid

    param = (low + high) / 2
    return cubic_bezier_point(param, 0, y1, y2, 1)


EASING_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    # Standard CSS
    "ease": (0.25, 0.1, 0.25, 1.0),
    "easeIn": (0.42, 0.0, 1.0, 1.0),
    "easeOut": (0.0, 0.0, 0.58, 1.0),
    "easeInOut": (0.42, 0.0, 0.58, 1.0),
    # Sine
    "easeInSine": (0.12, 0.0, 0.39, 0.0),
    "easeOutSine": (0.61, 1.0, 0.88, 1.0),
    "easeInOutSine": (0.37, 0.0, 0.63, 1.0),
    # Quad
    "easeInQuad": (0.11, 0.0, 0.5, 0.0),
    "easeOutQuad": (0.5, 1.0, 0.89, 1.0),
    "easeInOutQuad": (0.45, 0.0, 0.55, 1.0),
    # Cubic
    "easeInCubic": (0.32, 0.0, 0.67, 0.0),
    "easeOutCubic": (0.33, 1.0, 0.68, 1.0),
    "easeInOutCubic": (0.65, 0.0, 0.35, 1.0),
    # Quart
    "easeInQuart": (0.5, 0.0, 0.75, 0.0),
    "easeOutQuart": (0.25, 1.0, 0.5, 1.0),
    "easeInOutQuart": (0.76, 0.0, 0.24, 1.0),
    # Quint
    "easeInQuint": (0.64, 0.0, 0.78, 0.0),
    "easeOutQuint": (0.22, 1.0, 0.36, 1.0),
    "easeInOutQuint": (0.83, 0.0, 0.17, 1.0),
    # Expo
    "easeInExpo": (0.7, 0.0, 0.84, 0.0),
    "easeOutExpo": (0.16, 1.0, 0.3, 1.0),
    "easeInOutExpo": (0.87, 0.0, 0.13, 1.0),
    # Circ
    "easeInCirc": (0.55, 0.0, 1.0, 0.45),
    "easeOutCirc": (0.0, 0.55, 0.45, 1.0),
    "easeInOutCirc": (0.85, 0.0, 0.15, 1.0),
    # Back (overshoot)
    "easeInBack": (0.36, 0.0, 0.66, -0.56),
    "easeOutBack": (0.34, 1.56, 0.64, 1.0),
    "easeInOutBack": (0.68, -0.6, 0.32, 1.6),
    # Custom
    "snap": (0.0, 1.0, 0.0, 1.0),
    "anticipate": (0.38, -0.4, 0.88, 1.0),
    "overshoot": (0.25, 0.0, 0.0, 1.4),
    "bounce": (0.34, 1.2, 0.64, 1.0),
}


def apply_easing(t: float, easing_name: str) -> float:
    """Apply named easing function."""
    if easing_name not in EASING_PRESETS:
        return t
    x1, y1, x2, y2 = EASING_PRESETS[easing_name]
    return bezier_easing(x1, y1, x2, y2, t)


def apply_easing_to_range(
    t: float,
    from_val: float,
    to_val: float,
    easing_name: str = "linear"
) -> float:
    """Apply easing to interpolate between two values."""
    eased_t = apply_easing(t, easing_name)
    return from_val + (to_val - from_val) * eased_t


def list_easings() -> List[str]:
    """Get list of available easing names."""
    return sorted(EASING_PRESETS.keys())


__all__ = [
    "bezier_easing",
    "apply_easing",
    "apply_easing_to_range",
    "list_easings",
    "EASING_PRESETS",
]
