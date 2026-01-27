"""Koshi Animation Pipeline - Complete animation workflow node."""

import torch
from typing import Dict, List, Optional
from .core import apply_composite_transform


class KoshiAnimationPipeline:
    """Complete animation pipeline - generates multiple frames with motion."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Flux Motion"
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        """Generate animation frames with motion."""
        import comfy.sample
        import comfy.samplers
        import latent_preview

        frames = []
        prev_latent = None
        reference_image = None

        # Get motion frames if schedule provided
        motion_frames = None
        if motion_schedule is not None:
            motion_frames = motion_schedule.get("motion_frames", [])

        # Latent dimensions
        latent_height = height // 8
        latent_width = width // 8

        for frame_idx in range(num_frames):
            # Determine denoise strength
            denoise = denoise_first if frame_idx == 0 else denoise_rest

            # Get seed for this frame
            frame_seed = seed + frame_idx

            # Prepare latent
            if frame_idx == 0:
                if init_image is not None:
                    # Encode init image
                    latent = vae.encode(init_image[:, :, :, :3])
                else:
                    # Start from noise
                    latent = torch.zeros([1, 4, latent_height, latent_width])
            else:
                # Use previous frame's latent with motion applied
                latent = prev_latent.clone()

                # Apply motion transform if schedule provided
                if motion_frames and frame_idx < len(motion_frames):
                    mf = motion_frames[frame_idx]
                    motion_params = mf.to_dict()
                    latent = apply_composite_transform(latent, motion_params)

            # Sample
            samples = comfy.sample.sample(
                model,
                noise=comfy.sample.prepare_noise(latent, frame_seed, None),
                steps=steps,
                cfg=cfg,
                sampler_name="euler",
                scheduler="normal",
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=denoise,
            )

            # Decode to image
            image = vae.decode(samples)

            # Store for next iteration
            prev_latent = samples

            # Store reference for color matching
            if frame_idx == 0:
                reference_image = image

            frames.append(image)

        return (frames,)


class KoshiFrameIterator:
    """Iterate through frames for custom per-frame processing."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Flux Motion"
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
    "Koshi_AnimationPipeline": KoshiAnimationPipeline,
    "Koshi_FrameIterator": KoshiFrameIterator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_AnimationPipeline": "▄▀▄ KN Animation Pipeline",
    "Koshi_FrameIterator": "▄▀▄ KN Frame Iterator",
}
