"""Schedule parsing for Deforum-style keyframe strings."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import re

from .interpolation import interpolate_array


@dataclass
class MotionFrame:
    """Single frame's motion parameters."""
    frame_index: int
    zoom: float = 1.0
    angle: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    strength: float = 0.65
    prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for motion engine."""
        return {
            "zoom": self.zoom,
            "angle": self.angle,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
        }


# Default parameter values
DEFAULTS = {
    "zoom": 1.0,
    "angle": 0.0,
    "translation_x": 0.0,
    "translation_y": 0.0,
    "translation_z": 0.0,
    "strength": 0.65,
}


def parse_schedule_string(
    schedule: str,
    num_frames: int,
    default: float = 0.0,
    interpolation: str = "linear"
) -> List[float]:
    """
    Parse Deforum keyframe schedule string.

    Format: "frame:(value), frame:(value), ..."
    Examples:
        "0:(1.0), 30:(1.05), 60:(1.0)"
        "0:(0), 15:(-5), 30:(0)"

    Args:
        schedule: Keyframe schedule string
        num_frames: Total number of frames
        default: Default value for unspecified frames
        interpolation: "linear", "step", or "cubic"

    Returns:
        List of values, one per frame
    """
    if not schedule or not schedule.strip():
        return [default] * num_frames

    keyframes = _extract_keyframes(schedule)

    if not keyframes:
        return [default] * num_frames

    frames = sorted(keyframes.keys())
    values = [keyframes[f] for f in frames]

    return interpolate_array(frames, values, num_frames, interpolation).tolist()


def _extract_keyframes(schedule: str) -> Dict[int, float]:
    """Extract keyframe dict from schedule string."""
    keyframes = {}

    # Pattern: "frame:(value)" or "frame:value"
    pattern = r'(\d+)\s*:\s*\(?\s*([-+]?\d*\.?\d+)\s*\)?'
    matches = re.findall(pattern, schedule)

    for frame_str, value_str in matches:
        try:
            frame = int(frame_str)
            value = float(value_str)
            keyframes[frame] = value
        except ValueError:
            pass

    return keyframes


def parse_deforum_params(
    params: Dict[str, Any],
    num_frames: int,
    interpolation: str = "linear"
) -> List[MotionFrame]:
    """
    Convert Deforum parameters to motion frames.

    Args:
        params: Dictionary with zoom, angle, translation_x, etc.
                Values can be schedule strings or constants.
        num_frames: Total frames to generate
        interpolation: Interpolation method

    Returns:
        List of MotionFrame objects
    """
    def parse_param(param: Union[str, float, int], default: float) -> List[float]:
        if isinstance(param, str):
            return parse_schedule_string(param, num_frames, default, interpolation)
        elif isinstance(param, (int, float)):
            return [float(param)] * num_frames
        return [default] * num_frames

    zoom_values = parse_param(params.get("zoom", DEFAULTS["zoom"]), DEFAULTS["zoom"])
    angle_values = parse_param(params.get("angle", DEFAULTS["angle"]), DEFAULTS["angle"])
    tx_values = parse_param(params.get("translation_x", DEFAULTS["translation_x"]), DEFAULTS["translation_x"])
    ty_values = parse_param(params.get("translation_y", DEFAULTS["translation_y"]), DEFAULTS["translation_y"])
    tz_values = parse_param(params.get("translation_z", DEFAULTS["translation_z"]), DEFAULTS["translation_z"])
    strength_values = parse_param(params.get("strength", DEFAULTS["strength"]), DEFAULTS["strength"])

    # Parse prompts
    prompts = params.get("prompts", {})
    prompt_frames = _expand_prompts(prompts, num_frames)

    frames = []
    for i in range(num_frames):
        frames.append(MotionFrame(
            frame_index=i,
            zoom=zoom_values[i],
            angle=angle_values[i],
            translation_x=tx_values[i],
            translation_y=ty_values[i],
            translation_z=tz_values[i],
            strength=strength_values[i],
            prompt=prompt_frames.get(i),
        ))

    return frames


def _expand_prompts(prompts: Dict[int, str], num_frames: int) -> Dict[int, str]:
    """Expand keyframe prompts to per-frame mapping."""
    if not prompts:
        return {}

    result = {}
    sorted_frames = sorted(prompts.keys())

    current_prompt = None
    for i in range(num_frames):
        for kf in sorted_frames:
            if kf <= i:
                current_prompt = prompts[kf]
        if current_prompt:
            result[i] = current_prompt

    return result


__all__ = [
    "MotionFrame",
    "parse_schedule_string",
    "parse_deforum_params",
    "DEFAULTS",
]
