"""Koshi Motion Engine - Core motion processing for latents."""

import torch
from typing import Dict, Optional
from .core import apply_composite_transform


class KoshiMotionEngine:
    """Apply motion vectors and transformations to latents."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    SCHEDULE_PARAMS = {"zoom", "angle", "translation_x", "translation_y"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "translation_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "translation_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "motion_mask": ("MASK",),
                "motion_schedule": ("KOSHI_SCHEDULE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    def process(
        self,
        latent: Dict,
        zoom: float,
        angle: float,
        translation_x: float,
        translation_y: float,
        motion_mask: Optional[torch.Tensor] = None,
        motion_schedule: Optional[Dict] = None,
        frame_index: int = 0,
    ):
        """Apply motion transform to latent."""
        samples = latent["samples"].clone()

        zoom, angle, translation_x, translation_y = self._resolve_schedule(
            motion_schedule,
            frame_index,
            zoom,
            angle,
            translation_x,
            translation_y,
        )

        motion_params = {
            "zoom": zoom,
            "angle": angle,
            "translation_x": translation_x,
            "translation_y": translation_y,
        }

        # Apply transform
        transformed = apply_composite_transform(samples, motion_params)

        # Apply mask if provided (blend between original and transformed)
        if motion_mask is not None:
            mask = motion_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            # Resize mask to match latent spatial dims
            if mask.shape[-2:] != samples.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask, size=samples.shape[-2:], mode='bilinear', align_corners=False
                )

            mask = mask.expand_as(samples)
            transformed = samples * (1 - mask) + transformed * mask

        return ({"samples": transformed},)

    def _resolve_schedule(
        self,
        motion_schedule: Optional[Dict],
        frame_index: int,
        zoom: float,
        angle: float,
        translation_x: float,
        translation_y: float,
    ):
        """Apply a connected Koshi schedule to the matching motion parameter."""
        if motion_schedule is None:
            return zoom, angle, translation_x, translation_y

        # Legacy richer schedules are still accepted for old callers.
        if "motion_frames" in motion_schedule:
            frames = motion_schedule["motion_frames"]
            if frames:
                index = max(0, min(frame_index, len(frames) - 1))
                frame = frames[index]
                return frame.zoom, frame.angle, frame.translation_x, frame.translation_y

        values = motion_schedule.get("values")
        parameter = motion_schedule.get("name", "zoom")
        if not values or parameter not in self.SCHEDULE_PARAMS:
            return zoom, angle, translation_x, translation_y

        index = max(0, min(frame_index, len(values) - 1))
        value = float(values[index])

        if parameter == "zoom":
            zoom = value
        elif parameter == "angle":
            angle = value
        elif parameter == "translation_x":
            translation_x = value
        elif parameter == "translation_y":
            translation_y = value

        return zoom, angle, translation_x, translation_y


NODE_CLASS_MAPPINGS = {
    "Koshi_MotionEngine": KoshiMotionEngine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_MotionEngine": "▄▀▄ KN Motion Engine",
}
