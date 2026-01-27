"""Koshi Binary Node - Threshold and export for SIDKIT OLED displays."""
import torch
import numpy as np
from typing import Tuple


class KoshiBinary:
    """
    All-in-one binary conversion for SIDKIT OLED export.
    Threshold methods + optional hex output for C headers.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Image/SIDKIT"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("binary", "hex_data")
    OUTPUT_NODE = True

    METHODS = ["simple", "adaptive", "otsu", "dither_bayer", "dither_floyd"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (cls.METHODS, {"default": "simple"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "output_hex": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Adaptive options
                "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                "adaptive_c": ("FLOAT", {"default": 2.0, "min": -20.0, "max": 20.0, "step": 0.5}),
                # Target size for hex export
                "target_width": ("INT", {"default": 128, "min": 8, "max": 512, "step": 8}),
                "target_height": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
            }
        }

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3 and img.shape[2] >= 3:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img[:, :, 0] if len(img.shape) == 3 else img

    def _local_mean(self, gray: np.ndarray, block_size: int) -> np.ndarray:
        """Box filter for adaptive threshold."""
        try:
            from scipy.ndimage import uniform_filter
            return uniform_filter(gray.astype(np.float64), size=block_size, mode='reflect')
        except ImportError:
            # Cumsum fallback
            pad = block_size // 2
            padded = np.pad(gray, pad, mode='reflect')
            cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            h, w = gray.shape
            result = np.zeros_like(gray)
            for y in range(h):
                for x in range(w):
                    y1, y2 = y, y + block_size
                    x1, x2 = x, x + block_size
                    result[y, x] = (cumsum[y2, x2] - cumsum[y1, x2] - cumsum[y2, x1] + cumsum[y1, x1]) / (block_size ** 2)
            return result

    def _otsu_threshold(self, gray: np.ndarray) -> float:
        """Calculate Otsu's optimal threshold."""
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        hist = hist.astype(np.float32) / hist.sum()
        best_thresh, best_var = 0.5, 0
        for t in range(1, 255):
            w0, w1 = hist[:t].sum(), hist[t:].sum()
            if w0 == 0 or w1 == 0:
                continue
            m0 = (hist[:t] * np.arange(t)).sum() / w0
            m1 = (hist[t:] * np.arange(t, 256)).sum() / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var, best_thresh = var, t / 255.0
        return best_thresh

    def _bayer_matrix(self, n: int) -> np.ndarray:
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([[4*smaller, 4*smaller+2], [4*smaller+3, 4*smaller+1]]) / (n*n)

    def _pack_to_hex(self, binary: np.ndarray, width: int, height: int) -> str:
        """Pack binary to hex string for C headers."""
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((binary * 255).astype(np.uint8))
            pil = pil.resize((width, height), PILImage.NEAREST)
            binary = np.array(pil).astype(np.float32) / 255.0
            binary = (binary > 0.5).astype(np.uint8)
        except ImportError:
            pass

        h, w = binary.shape
        pad_w = (8 - w % 8) % 8
        if pad_w > 0:
            binary = np.pad(binary, ((0, 0), (0, pad_w)), constant_values=0)

        packed = np.zeros((h, binary.shape[1] // 8), dtype=np.uint8)
        for bit in range(8):
            packed |= (binary[:, bit::8] << bit)  # LSB first for XBM

        hex_vals = [f"0x{b:02x}" for b in packed.flatten()]
        return f"// {width}x{height} 1-bit\nconst uint8_t bitmap[{len(hex_vals)}] = {{\n  " + ", ".join(hex_vals) + "\n};"

    def convert(
        self,
        image: torch.Tensor,
        method: str,
        threshold: float,
        invert: bool,
        output_hex: bool,
        block_size: int = 11,
        adaptive_c: float = 2.0,
        target_width: int = 128,
        target_height: int = 64,
    ) -> Tuple[torch.Tensor, str]:
        results = []
        hex_outputs = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()
            gray = self._to_gray(img_np)

            if method == "simple":
                binary = (gray > threshold).astype(np.float32)
            elif method == "adaptive":
                local_mean = self._local_mean(gray, block_size)
                binary = (gray > (local_mean - adaptive_c / 255.0)).astype(np.float32)
            elif method == "otsu":
                auto_thresh = self._otsu_threshold(gray)
                binary = (gray > auto_thresh).astype(np.float32)
            elif method == "dither_bayer":
                bayer = self._bayer_matrix(4)
                h, w = gray.shape
                tiled = np.tile(bayer, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
                binary = (gray > tiled).astype(np.float32)
            elif method == "dither_floyd":
                result = gray.copy()
                h, w = gray.shape
                for y in range(h):
                    for x in range(w):
                        old = result[y, x]
                        new = 1.0 if old > threshold else 0.0
                        result[y, x] = new
                        err = old - new
                        if x + 1 < w: result[y, x+1] += err * 7/16
                        if y + 1 < h:
                            if x > 0: result[y+1, x-1] += err * 3/16
                            result[y+1, x] += err * 5/16
                            if x + 1 < w: result[y+1, x+1] += err * 1/16
                binary = (np.clip(result, 0, 1) > 0.5).astype(np.float32)
            else:
                binary = (gray > threshold).astype(np.float32)

            if invert:
                binary = 1.0 - binary

            result = np.stack([binary, binary, binary], axis=-1)
            results.append(torch.from_numpy(result.astype(np.float32)))

            if output_hex:
                hex_outputs.append(self._pack_to_hex(binary, target_width, target_height))

        hex_str = "\n\n".join(hex_outputs) if output_hex else ""
        return (torch.stack(results).to(image.device), hex_str)


NODE_CLASS_MAPPINGS = {
    "Koshi_Binary": KoshiBinary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Binary": "░▒░ KN Binary",
}
