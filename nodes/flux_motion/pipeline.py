"""Koshi Animation Pipeline - Complete animation workflow node."""

import torch
import warnings
from typing import Dict, List, Optional
from .core import apply_composite_transform


class KoshiAnimationPipeline:
    """
    [DEPRECATED] Use modular nodes with external KSampler instead.
    
    This node uses internal ComfyUI sampling APIs that are not stable.
    For animation workflows, use:
    
    KoshiScheduleMulti -> KoshiMotionEngine -> KSampler -> KoshiFeedback (loop)
    
    The modular approach works with FLUX and other models.
    """
    COLOR = "#4a1a1a"  # Reddish to indicate deprecated
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 30.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "motion_schedule": ("KOSHI_MOTION_SCHEDULE",),
                "init_image": ("IMAGE",),
                "denoise_first": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "denoise_rest": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "feedback_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def generate(
        self,
        model,
        clip,
        vae,
        positive,
        negative,
        num_frames: int,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        motion_schedule: Optional[Dict] = None,
        init_image: Optional[torch.Tensor] = None,
        denoise_first: float = 1.0,
        denoise_rest: float = 0.65,
        feedback_strength: float = 0.0,
    ):
        """[DEPRECATED] This node is deprecated. Use modular nodes with external KSampler."""
        warnings.warn(
            "KoshiAnimationPipeline is DEPRECATED. Use KoshiScheduleMulti + KoshiMotionEngine + "
            "external KSampler + KoshiFeedback for animation workflows. This node uses unstable internal APIs.",
            DeprecationWarning,
            stacklevel=2
        )
        
        print("[Koshi] WARNING: KoshiAnimationPipeline is deprecated and disabled.")
        print("[Koshi] Use modular workflow: KoshiScheduleMulti -> KoshiMotionEngine -> KSampler -> KoshiFeedback")
        
        # Return init_image or empty list
        if init_image is not None:
            return ([init_image],)
        return ([],)


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


# Deprecated: KoshiAnimationPipeline removed - use modular nodes with external KSampler

NODE_CLASS_MAPPINGS = {
    "Koshi_FrameIterator": KoshiFrameIterator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_FrameIterator": "▄▀▄ KN Frame Iterator",
}
