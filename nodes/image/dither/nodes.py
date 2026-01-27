"""Koshi Dither Node - All dithering techniques in one. SIDKIT Edition."""
import torch
import numpy as np
from typing import Tuple


class KoshiDither:
    """
    Universal dithering node - all algorithms in one.
    Perfect for SIDKIT OLED display export.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Image/SIDKIT"
    FUNCTION = "dither"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("dithered",)
    OUTPUT_NODE = True

    TECHNIQUES = ["bayer", "floyd_steinberg", "atkinson", "halftone", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "technique": (cls.TECHNIQUES, {"default": "bayer"}),
                "levels": ("INT", {"default": 2, "min": 2, "max": 256, "step": 1}),
                "grayscale": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Bayer options
                "bayer_size": (["2x2", "4x4", "8x8", "16x16"], {"default": "4x4"}),
                # Halftone options
                "dot_size": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 20.0, "step": 0.5}),
                "dot_angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0, "step": 5.0}),
                "dot_shape": (["circle", "square", "diamond"], {"default": "circle"}),
            }
        }

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(img.shape) == 3 and img.shape[2] >= 3:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img[:, :, 0] if len(img.shape) == 3 else img

    def _bayer_matrix(self, n: int) -> np.ndarray:
        """Generate Bayer dither matrix."""
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ]) / (n * n)

    def _dither_bayer(self, gray: np.ndarray, levels: int, size: int) -> np.ndarray:
        """Ordered Bayer dithering."""
        bayer = self._bayer_matrix(size)
        h, w = gray.shape
        tiled = np.tile(bayer, ((h + size - 1) // size, (w + size - 1) // size))[:h, :w]
        dithered = gray + (tiled - 0.5) / levels
        return np.clip(np.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)

    def _dither_floyd_steinberg(self, gray: np.ndarray, levels: int) -> np.ndarray:
        """Floyd-Steinberg error diffusion."""
        h, w = gray.shape
        result = gray.astype(np.float32).copy()
        for y in range(h):
            for x in range(w):
                old = result[y, x]
                new = np.round(old * (levels - 1)) / (levels - 1)
                result[y, x] = new
                err = old - new
                if x + 1 < w:
                    result[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        result[y + 1, x - 1] += err * 3 / 16
                    result[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        result[y + 1, x + 1] += err * 1 / 16
        return np.clip(result, 0, 1)

    def _dither_atkinson(self, gray: np.ndarray, levels: int) -> np.ndarray:
        """Atkinson dithering - lighter result, classic Mac style."""
        h, w = gray.shape
        result = gray.astype(np.float32).copy()
        threshold = 0.5 if levels == 2 else None
        for y in range(h):
            for x in range(w):
                old = result[y, x]
                if threshold is not None:
                    new = 1.0 if old > threshold else 0.0
                else:
                    new = np.round(old * (levels - 1)) / (levels - 1)
                result[y, x] = new
                err = (old - new) / 8  # Only 6/8 distributed
                if x + 1 < w:
                    result[y, x + 1] += err
                if x + 2 < w:
                    result[y, x + 2] += err
                if y + 1 < h:
                    if x > 0:
                        result[y + 1, x - 1] += err
                    result[y + 1, x] += err
                    if x + 1 < w:
                        result[y + 1, x + 1] += err
                if y + 2 < h:
                    result[y + 2, x] += err
        return np.clip(result, 0, 1)

    def _dither_halftone(self, gray: np.ndarray, dot_size: float, angle: float, shape: str) -> np.ndarray:
        """Halftone pattern dithering."""
        h, w = gray.shape
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        x_rot = x_coords * cos_t + y_coords * sin_t
        y_rot = -x_coords * sin_t + y_coords * cos_t
        cell_x = (x_rot / dot_size) % 1.0 - 0.5
        cell_y = (y_rot / dot_size) % 1.0 - 0.5
        center_x = (x_rot / dot_size).astype(int) * dot_size
        center_y = (y_rot / dot_size).astype(int) * dot_size
        orig_x = np.clip((center_x * cos_t - center_y * sin_t).astype(int), 0, w - 1)
        orig_y = np.clip((center_x * sin_t + center_y * cos_t).astype(int), 0, h - 1)
        intensity = gray[orig_y, orig_x]
        if shape == "circle":
            dist = np.sqrt(cell_x ** 2 + cell_y ** 2)
        elif shape == "square":
            dist = np.maximum(np.abs(cell_x), np.abs(cell_y))
        else:  # diamond
            dist = np.abs(cell_x) + np.abs(cell_y)
        radius = np.sqrt(1.0 - intensity) * 0.5
        return 1.0 - (dist < radius).astype(np.float32)

    def dither(
        self,
        image: torch.Tensor,
        technique: str,
        levels: int,
        grayscale: bool,
        bayer_size: str = "4x4",
        dot_size: float = 4.0,
        dot_angle: float = 45.0,
        dot_shape: str = "circle",
    ) -> Tuple[torch.Tensor]:
        results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()

            if grayscale or technique == "halftone":
                gray = self._to_gray(img_np)

                if technique == "bayer":
                    size = int(bayer_size.split("x")[0])
                    dithered = self._dither_bayer(gray, levels, size)
                elif technique == "floyd_steinberg":
                    dithered = self._dither_floyd_steinberg(gray, levels)
                elif technique == "atkinson":
                    dithered = self._dither_atkinson(gray, levels)
                elif technique == "halftone":
                    dithered = self._dither_halftone(gray, dot_size, dot_angle, dot_shape)
                else:  # none
                    dithered = np.floor(gray * (levels - 1) + 0.5) / (levels - 1)

                result = np.stack([dithered, dithered, dithered], axis=-1)
            else:
                # Color dithering - apply to each channel
                result = np.zeros_like(img_np)
                size = int(bayer_size.split("x")[0]) if technique == "bayer" else 4
                for c in range(3):
                    if technique == "bayer":
                        result[:, :, c] = self._dither_bayer(img_np[:, :, c], levels, size)
                    elif technique == "floyd_steinberg":
                        result[:, :, c] = self._dither_floyd_steinberg(img_np[:, :, c], levels)
                    elif technique == "atkinson":
                        result[:, :, c] = self._dither_atkinson(img_np[:, :, c], levels)
                    else:
                        result[:, :, c] = np.floor(img_np[:, :, c] * (levels - 1) + 0.5) / (levels - 1)

            results.append(torch.from_numpy(result.astype(np.float32)))

        return (torch.stack(results).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "Koshi_Dither": KoshiDither,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Dither": "░▒░ KN Dither",
}
