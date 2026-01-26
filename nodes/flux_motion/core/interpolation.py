"""Interpolation functions for keyframe animation."""

from typing import List
import numpy as np

try:
    from scipy.interpolate import CubicSpline, interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def linear_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """Linear interpolation between keyframe values."""
    if len(frames) == 0:
        return 0.0
    if len(frames) == 1:
        return values[0]

    if target_frame <= frames[0]:
        return values[0]
    if target_frame >= frames[-1]:
        return values[-1]

    for i in range(len(frames) - 1):
        if frames[i] <= target_frame <= frames[i + 1]:
            t = (target_frame - frames[i]) / (frames[i + 1] - frames[i])
            return values[i] + t * (values[i + 1] - values[i])

    return values[-1]


def step_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """Step interpolation - hold value until next keyframe."""
    if len(frames) == 0:
        return 0.0

    for i in range(len(frames) - 1, -1, -1):
        if frames[i] <= target_frame:
            return values[i]

    return values[0]


def cubic_spline_interpolation(
    frames: List[int],
    values: List[float],
    target_frame: int
) -> float:
    """Cubic spline interpolation for smooth curves."""
    if not SCIPY_AVAILABLE or len(frames) < 2:
        return linear_interpolation(frames, values, target_frame)

    cs = CubicSpline(frames, values, bc_type='natural')
    target_frame = max(frames[0], min(frames[-1], target_frame))
    return float(cs(target_frame))


def interpolate_array(
    frames: List[int],
    values: List[float],
    total_frames: int,
    method: str = "linear"
) -> np.ndarray:
    """Interpolate values across all frames."""
    if len(frames) == 0:
        return np.zeros(total_frames)
    if len(frames) == 1:
        return np.full(total_frames, values[0])

    result = np.zeros(total_frames)

    if method == "step":
        for f in range(total_frames):
            result[f] = step_interpolation(frames, values, f)
    elif method == "cubic" and SCIPY_AVAILABLE and len(frames) >= 2:
        cs = CubicSpline(frames, values, bc_type='natural')
        all_frames = np.arange(total_frames)
        clamped = np.clip(all_frames, frames[0], frames[-1])
        result = cs(clamped)
        result[:frames[0]] = values[0]
        if frames[-1] + 1 < total_frames:
            result[frames[-1] + 1:] = values[-1]
    else:
        if SCIPY_AVAILABLE:
            f = interp1d(frames, values, kind='linear',
                        bounds_error=False,
                        fill_value=(values[0], values[-1]))
            result = f(np.arange(total_frames))
        else:
            for i in range(total_frames):
                result[i] = linear_interpolation(frames, values, i)

    return result


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation."""
    if abs(edge1 - edge0) < 1e-10:
        return 1.0 if x >= edge0 else 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


__all__ = [
    "linear_interpolation",
    "step_interpolation",
    "cubic_spline_interpolation",
    "interpolate_array",
    "lerp",
    "smoothstep",
]
