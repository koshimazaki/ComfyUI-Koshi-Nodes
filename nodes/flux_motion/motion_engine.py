"""Koshi Motion Engine - Core motion processing for latents."""

import torch
from typing import Dict, Optional
from .core import apply_composite_transform


class KoshiMotionEngine:
    """Apply motion vectors and transformations to latents."""

    CATEGORY = "Koshi/Flux Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

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
                "motion_schedule": ("KOSHI_MOTION_SCHEDULE",),
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

        # Get motion params from schedule if provided
        if motion_schedule is not None and "motion_frames" in motion_schedule:
            frames = motion_schedule["motion_frames"]
            if 0 <= frame_index < len(frames):
                mf = frames[frame_index]
                zoom = mf.zoom
                angle = mf.angle
                translation_x = mf.translation_x
                translation_y = mf.translation_y

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


class KoshiMotionBatch:
    """Apply motion to a batch of latents using schedule."""

    CATEGORY = "Koshi/Flux Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "motion_schedule": ("KOSHI_MOTION_SCHEDULE",),
            },
            "optional": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    def process(
        self,
        latent: Dict,
        motion_schedule: Dict,
        start_frame: int = 0,
    ):
        """Apply motion transforms to all frames in batch."""
        samples = latent["samples"]
        batch_size = samples.shape[0]

        motion_frames = motion_schedule.get("motion_frames", [])
        results = []

        for i in range(batch_size):
            frame_idx = start_frame + i
            sample = samples[i:i+1]

            if frame_idx < len(motion_frames):
                mf = motion_frames[frame_idx]
                motion_params = mf.to_dict()
            else:
                motion_params = {"zoom": 1.0, "angle": 0.0, "translation_x": 0.0, "translation_y": 0.0}

            transformed = apply_composite_transform(sample, motion_params)
            results.append(transformed)

        output = torch.cat(results, dim=0)
        return ({"samples": output},)


NODE_CLASS_MAPPINGS = {
    "Koshi_MotionEngine": KoshiMotionEngine,
    "Koshi_MotionBatch": KoshiMotionBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_MotionEngine": "Koshi Motion Engine",
    "Koshi_MotionBatch": "Koshi Motion Batch",
}
