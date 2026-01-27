"""Koshi Greyscale Node - All greyscale conversions in one. SIDKIT Edition."""
import torch
import numpy as np
from typing import Tuple


class KoshiGreyscale:
    """
    Universal greyscale conversion with quantization for SIDKIT OLED displays.
    Supports multiple algorithms and bit depth output.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Image/SIDKIT"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("greyscale",)
    OUTPUT_NODE = True

    ALGORITHMS = ["luminosity", "average", "lightness", "red", "green", "blue"]
    BIT_DEPTHS = ["8-bit (256)", "4-bit (16)", "2-bit (4)", "1-bit (2)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": (cls.ALGORITHMS, {"default": "luminosity"}),
                "bit_depth": (cls.BIT_DEPTHS, {"default": "8-bit (256)"}),
                "dither": (["none", "bayer_2x2", "bayer_4x4", "bayer_8x8"], {"default": "none"}),
            },
            "optional": {
                "desaturate_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def _bayer_matrix(self, n: int) -> np.ndarray:
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ]) / (n * n)

    def convert(
        self,
        image: torch.Tensor,
        algorithm: str,
        bit_depth: str,
        dither: str,
        desaturate_amount: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        # Parse bit depth
        levels = {"8-bit (256)": 256, "4-bit (16)": 16, "2-bit (4)": 4, "1-bit (2)": 2}[bit_depth]

        results = []
        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()
            r, g, b_ch = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

            # Convert to greyscale based on algorithm
            if algorithm == "luminosity":
                gray = 0.299 * r + 0.587 * g + 0.114 * b_ch
            elif algorithm == "average":
                gray = (r + g + b_ch) / 3.0
            elif algorithm == "lightness":
                gray = (np.maximum(np.maximum(r, g), b_ch) + np.minimum(np.minimum(r, g), b_ch)) / 2.0
            elif algorithm == "red":
                gray = r
            elif algorithm == "green":
                gray = g
            elif algorithm == "blue":
                gray = b_ch
            else:
                gray = 0.299 * r + 0.587 * g + 0.114 * b_ch

            # Partial desaturation (blend with original)
            if desaturate_amount < 1.0:
                gray_rgb = np.stack([gray, gray, gray], axis=-1)
                result = img_np * (1 - desaturate_amount) + gray_rgb * desaturate_amount
                gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]

            # Apply dithering before quantization
            if dither != "none":
                n = int(dither.split("_")[1].split("x")[0])
                bayer = self._bayer_matrix(n)
                h, w = gray.shape
                tiled = np.tile(bayer, ((h + n - 1) // n, (w + n - 1) // n))[:h, :w]
                gray = gray + (tiled - 0.5) / levels

            # Quantize to bit depth
            gray = np.clip(gray, 0, 1)
            quantized = np.floor(gray * (levels - 1) + 0.5) / (levels - 1)

            result = np.stack([quantized, quantized, quantized], axis=-1)
            results.append(torch.from_numpy(result.astype(np.float32)))

        return (torch.stack(results).to(image.device),)


NODE_CLASS_MAPPINGS = {
    "Koshi_Greyscale": KoshiGreyscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Greyscale": "░▒░ KN Greyscale",
}
