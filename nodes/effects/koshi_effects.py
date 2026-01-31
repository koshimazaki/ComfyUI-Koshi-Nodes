"""
Koshi Effects - Unified effects node with swappable effect types.
Stack multiple nodes to combine effects. Each has live WebGL preview.

Effects: Dither, Bloom, Glitch, Hologram, Video Glitch, Scanlines, Chromatic Aberration
"""

import torch
import numpy as np
import os
import uuid
from typing import Tuple

try:
    from scipy.ndimage import gaussian_filter, sobel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from PIL import Image
    import folder_paths
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False


def save_preview(tensor):
    """Save tensor for ComfyUI preview."""
    if not PREVIEW_AVAILABLE:
        return []
    results = []
    output_dir = folder_paths.get_temp_directory()
    batch = tensor if len(tensor.shape) == 4 else tensor.unsqueeze(0)
    for i in range(batch.shape[0]):
        img_np = (np.clip(batch[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        filename = f"koshi_fx_{uuid.uuid4().hex[:8]}_{i}.png"
        Image.fromarray(img_np).save(os.path.join(output_dir, filename))
        results.append({"filename": filename, "subfolder": "", "type": "temp"})
    return results


class KoshiEffects:
    """
    Unified effects node - select effect type, adjust parameters, stack to combine.
    Live WebGL preview shows effect in real-time.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Effects"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    EFFECT_TYPES = [
        "dither",
        "bloom",
        "glitch",
        "hologram",
        "video_glitch",
        "scanlines",
        "chromatic",
    ]

    DITHER_METHODS = ["bayer", "floyd_steinberg", "atkinson", "halftone"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "effect_type": (cls.EFFECT_TYPES, {"default": "glitch"}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "display": "slider"}),
            },
            "optional": {
                # Dither params
                "dither_method": (cls.DITHER_METHODS, {"default": "bayer"}),
                "dither_levels": ("INT", {"default": 4, "min": 2, "max": 256, "step": 1}),

                # Bloom params
                "bloom_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bloom_radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Glitch params
                "rgb_shift": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "shake_amount": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "noise_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scan_lines": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Hologram params
                "holo_color": (["cyan", "green", "purple", "orange", "white"], {"default": "cyan"}),
                "edge_glow": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grid_opacity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),

                # Scanlines params
                "scanline_count": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "scanline_direction": (["horizontal", "vertical"], {"default": "horizontal"}),

                # Chromatic params
                "red_offset": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "blue_offset": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.1}),

                # Common
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    # ========== DITHER ==========
    def _bayer_matrix(self, n):
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ]) / (n * n)

    def _apply_dither(self, img, method, levels, intensity):
        h, w, c = img.shape
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        if method == "bayer":
            bayer = self._bayer_matrix(8)
            tiled = np.tile(bayer, ((h + 7) // 8, (w + 7) // 8))[:h, :w]
            dithered = gray + (tiled - 0.5) * intensity / max(1, levels - 1)
            dithered = np.clip(np.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)

        elif method == "floyd_steinberg":
            dithered = gray.copy()
            for y in range(h):
                for x in range(w):
                    old = dithered[y, x]
                    new = np.round(old * (levels - 1)) / (levels - 1)
                    dithered[y, x] = new
                    err = (old - new) * intensity
                    if x + 1 < w:
                        dithered[y, x + 1] += err * 7 / 16
                    if y + 1 < h:
                        if x > 0:
                            dithered[y + 1, x - 1] += err * 3 / 16
                        dithered[y + 1, x] += err * 5 / 16
                        if x + 1 < w:
                            dithered[y + 1, x + 1] += err * 1 / 16
            dithered = np.clip(dithered, 0, 1)

        elif method == "atkinson":
            dithered = gray.copy()
            for y in range(h):
                for x in range(w):
                    old = dithered[y, x]
                    new = np.round(old * (levels - 1)) / (levels - 1)
                    dithered[y, x] = new
                    err = (old - new) * intensity / 8
                    if x + 1 < w:
                        dithered[y, x + 1] += err
                    if x + 2 < w:
                        dithered[y, x + 2] += err
                    if y + 1 < h:
                        if x > 0:
                            dithered[y + 1, x - 1] += err
                        dithered[y + 1, x] += err
                        if x + 1 < w:
                            dithered[y + 1, x + 1] += err
                    if y + 2 < h:
                        dithered[y + 2, x] += err
            dithered = np.clip(dithered, 0, 1)

        else:  # halftone
            dot_size = max(2, int(10 * (1 - intensity) + 2))
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            cell_x = (x_coords / dot_size) % 1.0 - 0.5
            cell_y = (y_coords / dot_size) % 1.0 - 0.5
            dist = np.sqrt(cell_x ** 2 + cell_y ** 2)
            radius = np.sqrt(1.0 - gray) * 0.5
            dithered = 1.0 - (dist < radius).astype(np.float32)

        return np.stack([dithered] * 3, axis=-1)

    # ========== BLOOM ==========
    def _apply_bloom(self, img, threshold, intensity, radius):
        if not SCIPY_AVAILABLE:
            return img

        # Extract bright areas
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        bright_mask = np.maximum(luminance - threshold, 0)

        # Blur the bright areas
        sigma = radius * 50 + 1
        bloom = np.zeros_like(img)
        for c in range(3):
            bright = img[:, :, c] * bright_mask
            bloom[:, :, c] = gaussian_filter(bright, sigma=sigma)

        return np.clip(img + bloom * intensity, 0, 1)

    # ========== GLITCH ==========
    def _apply_glitch(self, img, intensity, rgb_shift, shake, noise, scan_lines, time, seed):
        np.random.seed(seed)
        h, w, c = img.shape
        result = img.copy()

        # Shake
        shake_x = int((np.random.rand() * 2 - 1) * shake * intensity)
        shake_y = int((np.random.rand() * 2 - 1) * shake * intensity)
        result = np.roll(np.roll(result, shake_x, axis=1), shake_y, axis=0)

        # RGB shift
        shift = int(rgb_shift * intensity)
        if shift > 0:
            r = np.roll(result[:, :, 0], shift, axis=1)
            b = np.roll(result[:, :, 2], -shift, axis=1)
            result[:, :, 0] = r
            result[:, :, 2] = b

        # White noise
        if noise > 0:
            noise_layer = (np.random.rand(h, w, 1) * 2 - 1) * noise * intensity
            result = np.clip(result + noise_layer, 0, 1)

        # Scan lines
        if scan_lines > 0:
            scanline = np.sin(np.arange(h) * 50)[:, np.newaxis, np.newaxis]
            result = result - (scanline * 0.5 + 0.5) * scan_lines * intensity * 0.3

        return np.clip(result, 0, 1)

    # ========== HOLOGRAM ==========
    def _apply_hologram(self, img, intensity, color, edge_glow, grid_opacity, scan_lines, time):
        h, w, c = img.shape
        colors = {
            "cyan": (0.0, 0.84, 0.99),
            "green": (0.0, 1.0, 0.4),
            "purple": (0.6, 0.2, 1.0),
            "orange": (1.0, 0.5, 0.0),
            "white": (1.0, 1.0, 1.0),
        }
        tint = colors.get(color, colors["cyan"])

        # Convert to grayscale and tint
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        result = np.stack([gray * tint[0], gray * tint[1], gray * tint[2]], axis=-1)

        # Edge glow
        if edge_glow > 0 and SCIPY_AVAILABLE:
            edge_x = sobel(gray, axis=1)
            edge_y = sobel(gray, axis=0)
            edges = np.sqrt(edge_x ** 2 + edge_y ** 2)
            edges = edges / (edges.max() + 1e-6)
            for i in range(3):
                result[:, :, i] = np.clip(result[:, :, i] + edges * edge_glow * tint[i], 0, 1)

        # Grid overlay
        if grid_opacity > 0:
            grid_size = 20
            y, x = np.mgrid[0:h, 0:w]
            grid = ((x % grid_size < 1) | (y % grid_size < 1)).astype(float) * grid_opacity
            for i in range(3):
                result[:, :, i] = np.clip(result[:, :, i] + grid * tint[i] * 0.5, 0, 1)

        # Scanlines
        if scan_lines > 0:
            scanline = np.sin(np.arange(h) * 3.14 * 2 * 50 / h + time)[:, np.newaxis]
            scanline = (scanline * 0.5 + 0.5) * scan_lines * intensity
            result = result * (1 - scanline[:, :, np.newaxis] * 0.3)

        return np.clip(result * intensity + img * (1 - intensity), 0, 1)

    # ========== VIDEO GLITCH ==========
    def _apply_video_glitch(self, img, intensity, time, seed):
        np.random.seed(seed + int(time * 10))
        h, w, c = img.shape
        result = img.copy()

        # Wave distortion
        y = np.arange(h)
        wave = (np.sin(y * 0.1 + time * 5) * intensity * 20).astype(int)

        for row in range(h):
            shift = wave[row] % w
            result[row] = np.roll(img[row], shift, axis=0)

        # RGB channel splitting
        if intensity > 0.3:
            for row in range(h):
                if np.random.rand() > 0.9:
                    offset = int((np.random.rand() - 0.5) * 30 * intensity)
                    result[row, :, 0] = np.roll(result[row, :, 0], offset)
                    result[row, :, 2] = np.roll(result[row, :, 2], -offset)

        return np.clip(result, 0, 1)

    # ========== SCANLINES ==========
    def _apply_scanlines(self, img, intensity, count, direction, time):
        h, w, c = img.shape

        if direction == "horizontal":
            scanline = np.sin(np.arange(h) * count / h * np.pi * 2 + time)[:, np.newaxis, np.newaxis]
        else:
            scanline = np.sin(np.arange(w) * count / w * np.pi * 2 + time)[np.newaxis, :, np.newaxis]

        scanline = (scanline * 0.5 + 0.5) * intensity
        return np.clip(img * (1 - scanline * 0.5), 0, 1)

    # ========== CHROMATIC ==========
    def _apply_chromatic(self, img, intensity, red_offset, blue_offset):
        h, w, c = img.shape
        r_shift = int(red_offset * intensity)
        b_shift = int(blue_offset * intensity)

        result = img.copy()
        result[:, :, 0] = np.roll(img[:, :, 0], r_shift, axis=1)
        result[:, :, 2] = np.roll(img[:, :, 2], b_shift, axis=1)

        return result

    # ========== MAIN APPLY ==========
    def apply(self, image, effect_type, intensity,
              dither_method="bayer", dither_levels=4,
              bloom_threshold=0.8, bloom_radius=0.5,
              rgb_shift=6.0, shake_amount=8.0, noise_amount=0.15, scan_lines=0.15,
              holo_color="cyan", edge_glow=0.5, grid_opacity=0.2,
              scanline_count=100, scanline_direction="horizontal",
              red_offset=1.0, blue_offset=-1.0,
              time=0.0, seed=0):

        results = []
        for b in range(image.shape[0]):
            img = image[b].cpu().numpy()

            if effect_type == "dither":
                out = self._apply_dither(img, dither_method, dither_levels, intensity)
            elif effect_type == "bloom":
                out = self._apply_bloom(img, bloom_threshold, intensity, bloom_radius)
            elif effect_type == "glitch":
                out = self._apply_glitch(img, intensity, rgb_shift, shake_amount,
                                          noise_amount, scan_lines, time, seed + b)
            elif effect_type == "hologram":
                out = self._apply_hologram(img, intensity, holo_color, edge_glow,
                                            grid_opacity, scan_lines, time)
            elif effect_type == "video_glitch":
                out = self._apply_video_glitch(img, intensity, time, seed + b)
            elif effect_type == "scanlines":
                out = self._apply_scanlines(img, intensity, scanline_count,
                                             scanline_direction, time)
            elif effect_type == "chromatic":
                out = self._apply_chromatic(img, intensity, red_offset, blue_offset)
            else:
                out = img

            results.append(torch.from_numpy(out.astype(np.float32)))

        output = torch.stack(results).to(image.device)
        preview = save_preview(output)

        return {"ui": {"images": preview}, "result": (output,)}


NODE_CLASS_MAPPINGS = {"Koshi_Effects": KoshiEffects}
NODE_DISPLAY_NAME_MAPPINGS = {"Koshi_Effects": "░▀░ Koshi Effects"}
