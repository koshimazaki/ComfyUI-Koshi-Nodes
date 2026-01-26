"""Core motion utilities for Koshi Flux Motion nodes."""

from .interpolation import (
    linear_interpolation,
    step_interpolation,
    cubic_spline_interpolation,
    interpolate_array,
    lerp,
    smoothstep,
)
from .easing import (
    apply_easing,
    apply_easing_to_range,
    EASING_PRESETS,
    list_easings,
)
from .transforms import (
    create_affine_matrix,
    apply_affine_transform,
    apply_composite_transform,
)
from .schedule_parser import (
    MotionFrame,
    parse_schedule_string,
    parse_deforum_params,
)

__all__ = [
    # Interpolation
    "linear_interpolation",
    "step_interpolation",
    "cubic_spline_interpolation",
    "interpolate_array",
    "lerp",
    "smoothstep",
    # Easing
    "apply_easing",
    "apply_easing_to_range",
    "EASING_PRESETS",
    "list_easings",
    # Transforms
    "create_affine_matrix",
    "apply_affine_transform",
    "apply_composite_transform",
    # Schedule
    "MotionFrame",
    "parse_schedule_string",
    "parse_deforum_params",
]
