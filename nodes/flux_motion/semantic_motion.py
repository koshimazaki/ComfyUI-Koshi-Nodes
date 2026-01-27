"""Koshi Semantic Motion - Generate motion from text descriptions."""

import re
from typing import Dict, List, Optional
from .core import MotionFrame


# Motion presets from natural language
MOTION_PRESETS = {
    # Zoom
    "zoom in": {"zoom_start": 1.0, "zoom_end": 1.1},
    "zoom out": {"zoom_start": 1.0, "zoom_end": 0.9},
    "slow zoom in": {"zoom_start": 1.0, "zoom_end": 1.05},
    "slow zoom out": {"zoom_start": 1.0, "zoom_end": 0.95},
    "fast zoom in": {"zoom_start": 1.0, "zoom_end": 1.2},
    "fast zoom out": {"zoom_start": 1.0, "zoom_end": 0.8},
    "dolly in": {"zoom_start": 1.0, "zoom_end": 1.15, "translation_z": 10},
    "dolly out": {"zoom_start": 1.0, "zoom_end": 0.85, "translation_z": -10},

    # Rotation
    "rotate left": {"angle_end": -15},
    "rotate right": {"angle_end": 15},
    "slow rotate left": {"angle_end": -5},
    "slow rotate right": {"angle_end": 5},
    "spin": {"angle_end": 360},
    "half spin": {"angle_end": 180},

    # Pan
    "pan left": {"translation_x_end": -30},
    "pan right": {"translation_x_end": 30},
    "pan up": {"translation_y_end": -30},
    "pan down": {"translation_y_end": 30},
    "slow pan left": {"translation_x_end": -15},
    "slow pan right": {"translation_x_end": 15},

    # Combined
    "orbit left": {"angle_end": -30, "zoom_start": 1.0, "zoom_end": 1.0},
    "orbit right": {"angle_end": 30, "zoom_start": 1.0, "zoom_end": 1.0},
    "push in": {"zoom_start": 1.0, "zoom_end": 1.15, "translation_y_end": -10},
    "pull out": {"zoom_start": 1.0, "zoom_end": 0.85, "translation_y_end": 10},

    # Static
    "static": {},
    "still": {},
    "hold": {},
}


class KoshiSemanticMotion:
    """Generate motion from semantic prompts (e.g., 'slow zoom in, pan left')."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Flux Motion"
    FUNCTION = "generate"
    RETURN_TYPES = ("KOSHI_MOTION_SCHEDULE",)
    RETURN_NAMES = ("motion_schedule",)

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(MOTION_PRESETS.keys())
        return {
            "required": {
                "motion_prompt": ("STRING", {
                    "multiline": True,
                    "default": "slow zoom in"
                }),
                "frames": ("INT", {"default": 60, "min": 1, "max": 1000}),
            },
            "optional": {
                "preset": (["from prompt"] + preset_list,),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "easing": (["linear", "easeInOut", "easeIn", "easeOut"],),
            }
        }

    def generate(
        self,
        motion_prompt: str,
        frames: int,
        preset: str = "from prompt",
        intensity: float = 1.0,
        easing: str = "linear",
    ):
        """Generate motion schedule from text description."""
        # Get motion params from preset or parse prompt
        if preset != "from prompt" and preset in MOTION_PRESETS:
            motion_params = MOTION_PRESETS[preset].copy()
        else:
            motion_params = self._parse_motion_prompt(motion_prompt)

        # Apply intensity multiplier
        for key in motion_params:
            if isinstance(motion_params[key], (int, float)):
                if key not in ["zoom_start", "zoom_end"]:
                    motion_params[key] *= intensity
                elif "zoom" in key:
                    # For zoom, intensity affects the delta from 1.0
                    delta = motion_params[key] - 1.0
                    motion_params[key] = 1.0 + delta * intensity

        # Generate motion frames
        motion_frames = self._generate_frames(motion_params, frames, easing)

        schedule = {
            "frames": frames,
            "motion_frames": motion_frames,
            "interpolation": "linear",
            "params": motion_params,
            "prompt": motion_prompt,
        }

        return (schedule,)

    def _parse_motion_prompt(self, prompt: str) -> Dict:
        """Parse natural language motion prompt."""
        prompt_lower = prompt.lower().strip()
        combined_params = {}

        # Check for each preset in the prompt
        for preset_name, params in MOTION_PRESETS.items():
            if preset_name in prompt_lower:
                for key, value in params.items():
                    if key not in combined_params:
                        combined_params[key] = value
                    elif isinstance(value, (int, float)):
                        combined_params[key] += value

        # Default to static if nothing matched
        if not combined_params:
            combined_params = {"zoom_start": 1.0, "zoom_end": 1.0}

        return combined_params

    def _generate_frames(
        self,
        params: Dict,
        num_frames: int,
        easing: str
    ) -> List[MotionFrame]:
        """Generate per-frame motion data."""
        from .core import apply_easing

        frames = []

        # Extract start/end values
        zoom_start = params.get("zoom_start", 1.0)
        zoom_end = params.get("zoom_end", zoom_start)
        angle_start = params.get("angle_start", 0.0)
        angle_end = params.get("angle_end", angle_start)
        tx_start = params.get("translation_x_start", 0.0)
        tx_end = params.get("translation_x_end", tx_start)
        ty_start = params.get("translation_y_start", 0.0)
        ty_end = params.get("translation_y_end", ty_start)

        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)

            # Apply easing
            if easing != "linear":
                t = apply_easing(t, easing)

            # Interpolate values
            zoom = zoom_start + t * (zoom_end - zoom_start)
            angle = angle_start + t * (angle_end - angle_start)
            tx = tx_start + t * (tx_end - tx_start)
            ty = ty_start + t * (ty_end - ty_start)

            frames.append(MotionFrame(
                frame_index=i,
                zoom=zoom,
                angle=angle,
                translation_x=tx,
                translation_y=ty,
            ))

        return frames


NODE_CLASS_MAPPINGS = {
    "Koshi_SemanticMotion": KoshiSemanticMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_SemanticMotion": "▄▀▄ KN Semantic Motion",
}
