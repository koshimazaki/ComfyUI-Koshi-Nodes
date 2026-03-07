"""Koshi Frame Iterator - Per-frame animation processing."""

import torch
from typing import Dict


class KoshiFrameIterator:
    """Iterate through frames for custom per-frame processing.
    
    Use this to build frame-by-frame animation loops with external samplers.
    Connect to KSampler for each frame, then collect results.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "iterate"
    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "FLOAT", "KOSHI_MOTION_SCHEDULE")
    RETURN_NAMES = ("image", "latent", "frame_index", "strength", "remaining_schedule")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "motion_schedule": ("KOSHI_MOTION_SCHEDULE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "vae": ("VAE",),
            }
        }

    def iterate(
        self,
        images: torch.Tensor,
        motion_schedule: Dict,
        frame_index: int,
        vae,
    ):
        """Get current frame and motion data for iteration."""
        batch_size = images.shape[0]

        # Clamp frame index
        frame_index = min(frame_index, batch_size - 1)

        # Get current frame
        image = images[frame_index:frame_index + 1]

        # Encode to latent
        latent = vae.encode(image[:, :, :, :3])

        # Get motion data
        motion_frames = motion_schedule.get("motion_frames", [])
        strength = 0.65
        if frame_index < len(motion_frames):
            strength = motion_frames[frame_index].strength

        return (image, {"samples": latent}, frame_index, strength, motion_schedule)


NODE_CLASS_MAPPINGS = {
    "Koshi_FrameIterator": KoshiFrameIterator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_FrameIterator": "▄▀▄ KN Frame Iterator",
}
