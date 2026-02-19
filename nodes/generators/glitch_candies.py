"""
Glitch Candies - Unified procedural pattern generator.
Combines 2D patterns, 3D raymarched shapes, morphing, and effects.
"""
import torch
import numpy as np

from .utils import save_preview
from .patterns_2d import PATTERNS_2D, generate_2d
from .patterns_3d import PATTERNS_3D, generate_3d, SDF_SHAPES
from .effects import (
    noise_displace, noise_overlay as apply_noise_overlay, apply_glitch_lines,
    apply_scanlines, apply_vignette, colorize
)


class KoshiGlitchCandies:
    """
    Unified procedural pattern generator with live WebGL preview.
    Generates 2D patterns, 3D raymarched shapes, with morphing and effects.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Generators"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True

    # All available patterns
    ALL_PATTERNS = list(PATTERNS_2D.keys()) + PATTERNS_3D

    # Shapes for morphing
    SHAPES = list(SDF_SHAPES.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "pattern": (cls.ALL_PATTERNS, {"default": "glitch_candies"}),
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                # Batch/Animation
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "loop_frames": ("INT", {"default": 0, "min": 0, "max": 1000,
                                        "tooltip": "Generate seamless loop (0 = disabled)"}),

                # 3D Camera (for rm_* patterns and shape_morph)
                "camera_distance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "rotation_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "rotation_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),

                # Shape Morphing (for shape_morph pattern)
                "shape_a": (cls.SHAPES, {"default": "sphere"}),
                "shape_b": (cls.SHAPES, {"default": "cube"}),
                "morph_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "display": "slider"}),

                # Noise Effects
                "noise_displacement": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "display": "slider"}),
                "noise_frequency": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5}),
                "noise_overlay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "display": "slider"}),

                # Post Effects
                "glitch_lines": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "display": "slider"}),
                "scanlines": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "display": "slider"}),
                "vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "display": "slider"}),

                # Color
                "color_mode": (["grayscale", "cyan", "green", "amber", "purple", "red"],
                               {"default": "grayscale"}),
            }
        }

    def generate(self, width, height, pattern, time, scale, seed,
                 batch_size=1, loop_frames=0,
                 camera_distance=3.0, rotation_x=0.0, rotation_y=0.0,
                 shape_a="sphere", shape_b="cube", morph_amount=0.0,
                 noise_displacement=0.0, noise_frequency=3.0, noise_overlay=0.0,
                 glitch_lines=0.0, scanlines=0.0, vignette=0.0,
                 color_mode="grayscale"):

        results = []

        # Determine frame count
        if loop_frames > 0:
            frame_count = loop_frames
            time_step = 10.0 / loop_frames  # 10 second loop
        else:
            frame_count = batch_size
            time_step = 0

        for i in range(frame_count):
            frame_time = time + (i * time_step if loop_frames > 0 else 0)
            frame_seed = seed + (i if loop_frames == 0 else 0)

            # Generate base pattern
            if pattern in PATTERNS_2D:
                frame = generate_2d(pattern, width, height, frame_time, scale, frame_seed)
            elif pattern.startswith("rm_") or pattern == "shape_morph":
                frame = generate_3d(
                    pattern, width, height, frame_time, scale, frame_seed,
                    camera_distance, rotation_x, rotation_y,
                    shape_a, shape_b, morph_amount,
                    noise_displacement, noise_frequency
                )
            else:
                frame = np.zeros((height, width))

            # Apply noise displacement (for 2D patterns)
            if noise_displacement > 0 and pattern in PATTERNS_2D:
                frame = noise_displace(frame, noise_displacement, noise_frequency * 10,
                                       4, frame_seed, frame_time)

            # Apply noise overlay
            if noise_overlay > 0:
                frame = apply_noise_overlay(frame, noise_overlay, noise_frequency, frame_time)

            # Apply glitch lines
            if glitch_lines > 0:
                frame = apply_glitch_lines(frame, glitch_lines, frame_time)

            # Colorize
            frame_rgb = colorize(frame, color_mode)

            # Apply scanlines (after color)
            if scanlines > 0:
                frame_rgb = apply_scanlines(frame_rgb, scanlines)

            # Apply vignette
            if vignette > 0:
                frame_rgb = apply_vignette(frame_rgb, vignette)

            results.append(frame_rgb.astype(np.float32))

        # Stack results
        rgb_stack = np.stack(results)
        grey_stack = np.mean(rgb_stack, axis=-1)

        image_tensor = torch.from_numpy(rgb_stack)
        mask_tensor = torch.from_numpy(grey_stack)

        # Return with preview
        preview_images = save_preview(image_tensor, "glitch_candies")
        return {
            "ui": {"images": preview_images},
            "result": (image_tensor, mask_tensor)
        }


# Keep utility nodes separate (they process input images)
class KoshiShapeMorph:
    """Blend between two input images with easing."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Generators"
    FUNCTION = "morph"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "display": "slider"}),
                "blend_mode": (["linear", "smooth", "ease_in", "ease_out", "sine"],),
            }
        }

    def morph(self, image_a, image_b, blend, blend_mode):
        from .effects import blend_images

        result = blend_images(image_a.numpy(), image_b.numpy(), blend, blend_mode)
        result = torch.from_numpy(result.astype(np.float32))

        mask = result[..., 0] * 0.299 + result[..., 1] * 0.587 + result[..., 2] * 0.114

        preview_images = save_preview(result, "morph")
        return {
            "ui": {"images": preview_images},
            "result": (result, mask)
        }


class KoshiNoiseDisplace:
    """Apply noise displacement to an image."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Generators"
    FUNCTION = "displace"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "display": "slider"}),
                "scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
            }
        }

    def displace(self, image, strength, scale, octaves, seed, time=0.0):
        results = []

        for b in range(image.shape[0]):
            img = image[b].cpu().numpy()
            displaced = noise_displace(img, strength, scale, octaves, seed + b, time)
            results.append(displaced)

        output_tensor = torch.from_numpy(np.stack(results).astype(np.float32))

        preview_images = save_preview(output_tensor, "displace")
        return {
            "ui": {"images": preview_images},
            "result": (output_tensor,)
        }
