"""Koshi Schedule Parser - Deforum-style keyframe scheduling."""

from typing import Dict, List, Any
from .core import parse_schedule_string, list_easings, EASING_PRESETS


class KoshiSchedule:
    """Parse Deforum-style schedule strings (e.g., '0:(1.0), 30:(0.5)')."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "parse"
    RETURN_TYPES = ("KOSHI_SCHEDULE",)
    RETURN_NAMES = ("schedule",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_string": ("STRING", {
                    "multiline": True,
                    "default": "0:(1.0), 30:(1.05), 60:(1.0)"
                }),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 10000}),
                "interpolation": (["linear", "cubic", "step"],),
                "easing": (["none"] + list_easings(),),
            },
            "optional": {
                "parameter_name": ("STRING", {"default": "zoom"}),
            }
        }

    def parse(
        self,
        schedule_string: str,
        max_frames: int,
        interpolation: str,
        easing: str,
        parameter_name: str = "zoom"
    ):
        """Parse schedule string into per-frame values."""
        values = parse_schedule_string(
            schedule_string,
            max_frames,
            default=1.0 if parameter_name == "zoom" else 0.0,
            interpolation=interpolation
        )

        # Apply easing to remap the interpolated values between keyframe endpoints.
        # This curves the transition timing without changing start/end values.
        if easing != "none" and easing in EASING_PRESETS and len(values) > 1:
            from .core import apply_easing
            first_val = values[0]
            last_val = values[-1]
            val_range = last_val - first_val
            if abs(val_range) > 1e-9:
                for i in range(len(values)):
                    t = i / (len(values) - 1)
                    eased_t = apply_easing(t, easing)
                    values[i] = first_val + eased_t * val_range

        schedule = {
            "name": parameter_name,
            "frames": max_frames,
            "values": values,
            "interpolation": interpolation,
            "easing": easing,
            "raw": schedule_string,
        }

        return (schedule,)


class KoshiScheduleMulti:
    """Parse multiple Deforum-style schedules for full motion control."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "parse"
    RETURN_TYPES = ("KOSHI_MOTION_SCHEDULE",)
    RETURN_NAMES = ("motion_schedule",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 10000}),
                "interpolation": (["linear", "cubic", "step"],),
            },
            "optional": {
                "zoom": ("STRING", {
                    "multiline": True,
                    "default": "0:(1.0)"
                }),
                "angle": ("STRING", {
                    "multiline": True,
                    "default": "0:(0)"
                }),
                "translation_x": ("STRING", {
                    "multiline": True,
                    "default": "0:(0)"
                }),
                "translation_y": ("STRING", {
                    "multiline": True,
                    "default": "0:(0)"
                }),
                "strength": ("STRING", {
                    "multiline": True,
                    "default": "0:(0.65)"
                }),
            }
        }

    def parse(
        self,
        max_frames: int,
        interpolation: str,
        zoom: str = "0:(1.0)",
        angle: str = "0:(0)",
        translation_x: str = "0:(0)",
        translation_y: str = "0:(0)",
        strength: str = "0:(0.65)",
    ):
        """Parse all motion schedules into unified motion data."""
        from .core import parse_deforum_params

        params = {
            "zoom": zoom,
            "angle": angle,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "strength": strength,
        }

        motion_frames = parse_deforum_params(params, max_frames, interpolation)

        schedule = {
            "frames": max_frames,
            "motion_frames": motion_frames,
            "interpolation": interpolation,
            "params": params,
        }

        return (schedule,)


NODE_CLASS_MAPPINGS = {
    "Koshi_Schedule": KoshiSchedule,
    "Koshi_ScheduleMulti": KoshiScheduleMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Schedule": "▀▄▀ KN Schedule Parser",
    "Koshi_ScheduleMulti": "▀▄▀ KN Multi-Schedule",
}
