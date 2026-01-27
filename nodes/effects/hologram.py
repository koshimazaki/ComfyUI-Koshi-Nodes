"""
Hologram Effect for ComfyUI
Based on alien.js hologram and koshimazaki/CreaturesSite HologramOG
Combines: scanlines, glitch, fresnel edge glow, wireframe grid, color tint
"""

import torch
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import sobel, gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KoshiHologram:
    """
    Apply hologram post-process effect to any image.
    Combines scanlines, glitch distortion, edge glow, grid overlay, and color tinting.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Effects"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hologram",)
    OUTPUT_NODE = True

    # Preset colors
    COLORS = {
        "cyan": (0.012, 0.843, 0.988),      # #03d7fc
        "red_error": (1.0, 0.0, 0.0),        # Error state
        "green_matrix": (0.0, 1.0, 0.4),     # Matrix style
        "purple": (0.6, 0.2, 1.0),           # Sci-fi purple
        "orange": (1.0, 0.5, 0.0),           # Warning
        "white": (1.0, 1.0, 1.0),            # Clean
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_preset": (list(cls.COLORS.keys()), {"default": "cyan"}),
                "scanline_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scanline_count": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "glitch_intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "edge_glow": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grid_opacity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grid_size": ("INT", {"default": 20, "min": 5, "max": 100, "step": 5}),
                "alpha": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "animate": ("BOOLEAN", {"default": False}),
                "background": (["black", "transparent", "original"],),
            }
        }

    def _apply_scanlines(self, img: np.ndarray, intensity: float, count: int, time: float) -> np.ndarray:
        """Add horizontal scanlines with time-based flicker."""
        h, w = img.shape[:2]
        y = np.arange(h).reshape(-1, 1)

        # Scanline pattern: sin wave with time animation
        scanlines = np.sin(y * (count / h) * np.pi * 2 + time * 5.0) * 0.5 + 0.5
        scanlines = 1.0 - (scanlines * intensity)

        # Apply to all channels
        return img * scanlines[:, :, np.newaxis] if len(img.shape) == 3 else img * scanlines

    def _apply_glitch(self, img: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """Apply horizontal glitch distortion."""
        if intensity <= 0:
            return img

        h, w = img.shape[:2]
        result = img.copy()

        # Create UV distortion based on Y position
        y = np.arange(h)
        offset = (np.sin(y * 50.0 + time * 2.0) * intensity * w).astype(int)

        # Random glitch bands
        np.random.seed(int(time * 10) % 1000)
        glitch_bands = np.random.rand(h) > 0.95
        offset = np.where(glitch_bands, offset * 3, offset)

        # Apply horizontal shift per row
        for row in range(h):
            shift = offset[row] % w
            if shift != 0:
                result[row] = np.roll(img[row], shift, axis=0)

        return result

    def _detect_edges(self, img: np.ndarray) -> np.ndarray:
        """Detect edges for fresnel-like glow effect."""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img

        if SCIPY_AVAILABLE:
            # Sobel edge detection
            edge_x = sobel(gray, axis=1)
            edge_y = sobel(gray, axis=0)
            edges = np.sqrt(edge_x**2 + edge_y**2)
            edges = gaussian_filter(edges, sigma=1.5)
        else:
            # Simple gradient fallback
            edge_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            edge_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            edges = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize
        edges = edges / (edges.max() + 1e-6)
        return edges

    def _apply_edge_glow(self, img: np.ndarray, edges: np.ndarray, intensity: float, color: Tuple[float, float, float]) -> np.ndarray:
        """Apply fresnel-like edge glow."""
        if intensity <= 0:
            return img

        # Boost edges with power function (like fresnel)
        glow = np.power(edges, 0.5) * intensity

        # Add colored glow
        result = img.copy()
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] + glow * color[c], 0, 1)

        return result

    def _apply_grid(self, img: np.ndarray, opacity: float, grid_size: int, color: Tuple[float, float, float]) -> np.ndarray:
        """Overlay wireframe grid pattern."""
        if opacity <= 0:
            return img

        h, w = img.shape[:2]
        result = img.copy()

        # Create grid pattern
        grid = np.zeros((h, w), dtype=np.float32)
        thickness = max(1, grid_size // 10)

        # Horizontal and vertical lines
        for y in range(0, h, grid_size):
            grid[max(0, y-thickness//2):min(h, y+thickness//2+1), :] = 1.0
        for x in range(0, w, grid_size):
            grid[:, max(0, x-thickness//2):min(w, x+thickness//2+1)] = 1.0

        # Blend grid with color
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - grid * opacity) + grid * opacity * color[c]

        return result

    def _apply_color_tint(self, img: np.ndarray, color: Tuple[float, float, float], strength: float = 0.7) -> np.ndarray:
        """Apply hologram color tint."""
        # Convert to grayscale luminance
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Tint with hologram color
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = lum * color[c] * strength + img[:, :, c] * (1 - strength)

        return np.clip(result, 0, 1)

    def apply(
        self,
        image: torch.Tensor,
        color_preset: str,
        scanline_intensity: float,
        scanline_count: int,
        glitch_intensity: float,
        edge_glow: float,
        grid_opacity: float,
        grid_size: int,
        alpha: float,
        time: float,
        animate: bool = False,
        background: str = "black",
    ) -> Tuple[torch.Tensor]:

        color = self.COLORS.get(color_preset, self.COLORS["cyan"])
        results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy().copy()

            # Animate time per frame if enabled
            frame_time = time + (b * 0.1) if animate else time

            # Store original for background option
            original = img_np.copy()

            # 1. Detect edges for glow
            edges = self._detect_edges(img_np)

            # 2. Apply color tint
            img_np = self._apply_color_tint(img_np, color, strength=0.8)

            # 3. Apply glitch distortion
            img_np = self._apply_glitch(img_np, glitch_intensity, frame_time)

            # 4. Apply scanlines
            img_np = self._apply_scanlines(img_np, scanline_intensity, scanline_count, frame_time)

            # 5. Apply edge glow
            img_np = self._apply_edge_glow(img_np, edges, edge_glow, color)

            # 6. Apply grid overlay
            img_np = self._apply_grid(img_np, grid_opacity, grid_size, color)

            # 7. Apply alpha/background
            if background == "black":
                img_np = img_np * alpha
            elif background == "original":
                img_np = original * (1 - alpha) + img_np * alpha
            # transparent keeps as-is with alpha in mind

            results.append(torch.from_numpy(img_np.astype(np.float32)))

        return (torch.stack(results).to(image.device),)


class KoshiScanlines:
    """Simple scanlines effect - can be used standalone or with other effects."""

    CATEGORY = "Koshi/Effects"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "direction": (["horizontal", "vertical"],),
                "animate_speed": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    def apply(self, image: torch.Tensor, count: int, intensity: float, direction: str, animate_speed: float) -> Tuple[torch.Tensor]:
        results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()
            h, w = img_np.shape[:2]

            time = b * animate_speed

            if direction == "horizontal":
                y = np.arange(h).reshape(-1, 1)
                pattern = np.sin(y * (count / h) * np.pi * 2 + time) * 0.5 + 0.5
                pattern = 1.0 - (pattern * intensity)
                result = img_np * pattern[:, :, np.newaxis]
            else:
                x = np.arange(w).reshape(1, -1)
                pattern = np.sin(x * (count / w) * np.pi * 2 + time) * 0.5 + 0.5
                pattern = 1.0 - (pattern * intensity)
                result = img_np * pattern[:, :, np.newaxis]

            results.append(torch.from_numpy(result.astype(np.float32)))

        return (torch.stack(results).to(image.device),)


class KoshiVideoGlitch:
    """Video glitch distortion effect."""

    CATEGORY = "Koshi/Effects"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("glitched",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "distortion": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "distortion2": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            }
        }

    def apply(self, image: torch.Tensor, distortion: float, distortion2: float, speed: float, seed: int) -> Tuple[torch.Tensor]:
        results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()
            h, w = img_np.shape[:2]
            result = img_np.copy()

            time = b * speed
            np.random.seed(seed + b)

            # Primary wave distortion
            y = np.arange(h)
            offset1 = (np.sin(y * 50.0 + time * 2.0) * distortion * w).astype(int)

            # Secondary random glitch bands
            glitch_mask = np.random.rand(h) > (1 - distortion2)
            offset2 = (np.random.rand(h) * distortion2 * w * 2 - distortion2 * w).astype(int)

            # Combine offsets
            offset = np.where(glitch_mask, offset1 + offset2, offset1)

            # Apply RGB split on glitch bands
            for row in range(h):
                shift = offset[row] % w
                if shift != 0:
                    result[row, :, 0] = np.roll(img_np[row, :, 0], shift)
                    result[row, :, 1] = np.roll(img_np[row, :, 1], shift // 2)
                    result[row, :, 2] = np.roll(img_np[row, :, 2], -shift // 3)

            results.append(torch.from_numpy(result.astype(np.float32)))

        return (torch.stack(results).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "Koshi_Hologram": KoshiHologram,
    "Koshi_Scanlines": KoshiScanlines,
    "Koshi_VideoGlitch": KoshiVideoGlitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Hologram": "Koshi Hologram",
    "Koshi_Scanlines": "Koshi Scanlines",
    "Koshi_VideoGlitch": "Koshi Video Glitch",
}
