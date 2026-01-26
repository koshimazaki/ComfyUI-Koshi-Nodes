"""
Glitch Shader Node for ComfyUI
Implements WebGL-style glitch effects with controllable parameters

Inspired by the work of Yoichi Kobayashi
Original WebGL shader: https://codepen.io/ykob/pen/GmEzoQ
Adapted for ComfyUI with additional control parameters
"""

import numpy as np
import torch
from PIL import Image
import math


class GlitchShaderNode:
    """
    A ComfyUI node that applies glitch shader effects to images.
    Based on WebGL shader with RGB shift, noise, and distortion effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "glitch_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "rgb_shift": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shake_amount": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider"
                }),
                "noise_amount": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "block_noise_size": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "scan_line_intensity": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "freeze": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Frozen",
                    "label_off": "Animated"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glitch"
    CATEGORY = "Koshi/Effects"

    def __init__(self):
        self.frozen_time = 0.0

    @staticmethod
    def simplex_noise_3d(x, y, z):
        """
        Simplified 3D simplex noise implementation
        """
        # Simple pseudo-noise based on sin/cos functions
        n = np.sin(x * 12.9898 + y * 78.233 + z * 45.164) * 43758.5453
        return np.fmod(n, 1.0) * 2.0 - 1.0

    @staticmethod
    def random_2d(x, y):
        """
        2D pseudo-random function matching GLSL random()
        """
        return np.fmod(np.sin(x * 12.9898 + y * 78.233) * 43758.5453, 1.0)

    def apply_glitch(self, image, time, glitch_intensity, rgb_shift, shake_amount,
                     noise_amount, block_noise_size, scan_line_intensity, freeze, seed):
        """
        Apply glitch shader effect to the input image
        """
        # Input validation
        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError("image must be a valid torch.Tensor")

        if len(image.shape) != 4:
            raise ValueError(f"image must be 4D tensor (batch, height, width, channels), got shape {image.shape}")

        if image.shape[0] == 0:
            raise ValueError("image batch cannot be empty")

        # Clamp parameters to safe ranges
        glitch_intensity = max(0.0, min(2.0, float(glitch_intensity)))
        rgb_shift = max(0.0, float(rgb_shift))
        shake_amount = max(0.0, float(shake_amount))
        noise_amount = max(0.0, min(1.0, float(noise_amount)))
        block_noise_size = max(1.0, float(block_noise_size))
        scan_line_intensity = max(0.0, min(1.0, float(scan_line_intensity)))

        # Set random seed for reproducibility
        np.random.seed(int(seed) & 0xFFFFFFFF)  # Ensure seed is valid 32-bit int

        # Handle freeze functionality
        if freeze:
            time = self.frozen_time
        else:
            self.frozen_time = time

        # Convert tensor to numpy array
        # ComfyUI images are in format [batch, height, width, channels]
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img_np = image[b].cpu().numpy()
            height, width, channels = img_np.shape

            # Create output array
            output = np.zeros_like(img_np)

            # Calculate glitch strength based on time interval
            interval = 3.0
            strength = self._smoothstep(interval * 0.5, interval,
                                       interval - np.fmod(time, interval))
            strength *= glitch_intensity

            # Apply shake effect
            shake_x = int((self.random_2d(time, 0) * 2.0 - 1.0) *
                         (shake_amount * strength + 0.5))
            shake_y = int((self.random_2d(time * 2.0, 0) * 2.0 - 1.0) *
                         (shake_amount * strength + 0.5))

            # VECTORIZED: Pre-compute all row-based values
            y_coords = np.arange(height)

            # Vectorized noise calculation for all rows
            noise_val1 = self.simplex_noise_3d(0, y_coords * 0.01, time * 400.0)
            noise_val2 = self.simplex_noise_3d(0, y_coords * 0.02, time * 200.0)

            # Vectorized rgb_wave calculation
            rgb_wave = (noise_val1 * (2.0 + strength * 32.0) *
                       noise_val2 * (1.0 + strength * 4.0))

            # Vectorized periodic spikes
            sin_vals1 = np.sin(y_coords * 0.005 + time * 1.6)
            sin_vals2 = np.sin(y_coords * 0.005 + time * 2.0)
            rgb_wave += (sin_vals1 > 0.9995).astype(float) * 12.0
            rgb_wave += (sin_vals2 > 0.9999).astype(float) * -18.0

            # Vectorized RGB difference
            rgb_diff = (rgb_shift + np.sin(time * 500.0 + y_coords / height * 40.0) *
                       (20.0 * strength + 1.0))

            rgb_wave_pixels = rgb_wave.astype(int)
            rgb_diff_pixels = rgb_diff.astype(int)

            # Create coordinate meshes for vectorized indexing
            y_mesh, x_mesh = np.meshgrid(y_coords, np.arange(width), indexing='ij')

            # Apply shake offset (vectorized)
            src_x = (x_mesh + shake_x) % width
            src_y = (y_mesh + shake_y) % height

            # RGB shift effect (vectorized with broadcasting)
            rgb_wave_broadcast = rgb_wave_pixels[:, np.newaxis]  # Shape: (height, 1)
            rgb_diff_broadcast = rgb_diff_pixels[:, np.newaxis]  # Shape: (height, 1)

            r_x = (src_x + rgb_wave_broadcast + rgb_diff_broadcast) % width
            g_x = (src_x + rgb_wave_broadcast) % width
            b_x = (src_x + rgb_wave_broadcast - rgb_diff_broadcast) % width

            # Vectorized channel assignment using fancy indexing
            output[:, :, 0] = img_np[src_y, r_x, 0]  # R channel
            if channels >= 2:
                output[:, :, 1] = img_np[src_y, g_x, 1]  # G channel
            if channels >= 3:
                output[:, :, 2] = img_np[src_y, b_x, 2]  # B channel
            if channels == 4:
                output[:, :, 3] = img_np[src_y, src_x, 3]  # Alpha

            # Apply white noise
            white_noise = (np.random.rand(height, width, 1) * 2.0 - 1.0) * \
                         (noise_amount + strength * noise_amount)
            output = np.clip(output + white_noise, 0.0, 1.0)

            # Apply block noise glitches
            output = self._apply_block_noise(output, time, strength,
                                            block_noise_size, rgb_shift, img_np)

            # Apply scan line effect
            scan_lines = np.sin(np.arange(height)[:, np.newaxis] * 1200.0 / height * 2 * np.pi)
            scan_lines = (scan_lines + 1.0) / 2.0 * (scan_line_intensity + strength * 0.2)
            output = np.clip(output - scan_lines[:, :, np.newaxis], 0.0, 1.0)

            results.append(output)

        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(results)).float()

        return (output_tensor,)

    def _smoothstep(self, edge0, edge1, x):
        """GLSL smoothstep function"""
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _apply_block_noise(self, output, time, strength, block_size, rgb_shift, original):
        """Apply block noise glitch effect"""
        height, width, channels = output.shape

        # First block noise layer
        bn_time = np.floor(time * 20.0) * 200.0

        for y in range(0, height, int(block_size * 10)):
            for x in range(0, width, int(block_size * 10)):
                noise_x = self.simplex_noise_3d(0, x / width * 3.0, bn_time)
                noise_y = self.simplex_noise_3d(0, y / height * 3.0, bn_time)

                mask_x = 1.0 if (noise_x + 1.0) / 2.0 < (0.12 + strength * 0.3) else 0.0
                mask_y = 1.0 if (noise_y + 1.0) / 2.0 < (0.12 + strength * 0.3) else 0.0
                mask = mask_x * mask_y

                if mask > 0.5:
                    # Apply block glitch
                    block_h = min(int(block_size * 10), height - y)
                    block_w = min(int(block_size * 10), width - x)

                    offset = int(np.sin(bn_time) * 20)
                    for dy in range(block_h):
                        for dx in range(block_w):
                            src_x = (x + dx + offset) % width
                            if y + dy < height and x + dx < width:
                                output[y + dy, x + dx] = original[y + dy, src_x]

        # Second block noise layer (different timing)
        bn_time2 = np.floor(time * 25.0) * 300.0

        for y in range(0, height, int(block_size * 5)):
            for x in range(0, width, int(block_size * 15)):
                noise_x = self.simplex_noise_3d(0, x / width * 2.0, bn_time2)
                noise_y = self.simplex_noise_3d(0, y / height * 8.0, bn_time2)

                mask_x = 1.0 if (noise_x + 1.0) / 2.0 < (0.12 + strength * 0.5) else 0.0
                mask_y = 1.0 if (noise_y + 1.0) / 2.0 < (0.12 + strength * 0.3) else 0.0
                mask = mask_x * mask_y

                if mask > 0.5:
                    # Apply second block glitch
                    block_h = min(int(block_size * 5), height - y)
                    block_w = min(int(block_size * 15), width - x)

                    offset = int(np.sin(bn_time2) * 15)
                    for dy in range(block_h):
                        for dx in range(block_w):
                            src_x = (x + dx + offset) % width
                            if y + dy < height and x + dx < width:
                                output[y + dy, x + dx] = original[y + dy, src_x]

        return output


# Node registration
NODE_CLASS_MAPPINGS = {
    "KoshiGlitchShader": GlitchShaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoshiGlitchShader": "Glitch Shader Effect"
}
