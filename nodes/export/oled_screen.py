"""OLED Screen Emulation and Sprite Sheet Generation."""
import torch
import numpy as np
from typing import Tuple

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ..utils.preview import save_images_for_preview


class KoshiSpriteSheet:
    """
    Combine frame sequence into a sprite sheet grid.
    Perfect for game dev exports and embedded display animations.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "create_sheet"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("sprite_sheet", "cols", "rows", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "layout": (["auto", "horizontal", "vertical", "grid"],),
                "max_cols": ("INT", {"default": 8, "min": 1, "max": 32}),
                "max_rows": ("INT", {"default": 8, "min": 1, "max": 32}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 16}),
                "background": (["transparent", "black", "white", "magenta"],),
            }
        }

    def create_sheet(
        self,
        images: torch.Tensor,
        layout: str,
        max_cols: int,
        max_rows: int,
        padding: int,
        background: str,
    ) -> Tuple[torch.Tensor, int, int, int]:
        batch_size = images.shape[0]
        h, w, c = images.shape[1], images.shape[2], images.shape[3]

        # Calculate grid dimensions
        if layout == "horizontal":
            cols = batch_size
            rows = 1
        elif layout == "vertical":
            cols = 1
            rows = batch_size
        elif layout == "grid":
            cols = min(max_cols, batch_size)
            rows = (batch_size + cols - 1) // cols
        else:  # auto - try to make roughly square
            cols = int(np.ceil(np.sqrt(batch_size)))
            rows = int(np.ceil(batch_size / cols))

        # Ensure within limits
        cols = min(cols, max_cols)
        rows = min(rows, max_rows)

        # Calculate sheet dimensions
        sheet_w = cols * (w + padding) - padding
        sheet_h = rows * (h + padding) - padding

        # Background color
        bg_colors = {
            "transparent": [0.0, 0.0, 0.0, 0.0],
            "black": [0.0, 0.0, 0.0],
            "white": [1.0, 1.0, 1.0],
            "magenta": [1.0, 0.0, 1.0],
        }
        bg = bg_colors.get(background, [0.0, 0.0, 0.0])

        # Create sheet (add alpha channel if transparent)
        if background == "transparent":
            sheet = torch.zeros((sheet_h, sheet_w, 4), dtype=images.dtype, device=images.device)
        else:
            sheet = torch.zeros((sheet_h, sheet_w, c), dtype=images.dtype, device=images.device)
            for i, v in enumerate(bg[:c]):
                sheet[:, :, i] = v

        # Place frames
        for idx in range(min(batch_size, cols * rows)):
            row = idx // cols
            col = idx % cols
            y = row * (h + padding)
            x = col * (w + padding)

            if background == "transparent":
                # Add alpha = 1 for image areas
                sheet[y:y+h, x:x+w, :c] = images[idx]
                sheet[y:y+h, x:x+w, 3] = 1.0
            else:
                sheet[y:y+h, x:x+w] = images[idx]

        return (sheet.unsqueeze(0), cols, rows, batch_size)


class KoshiOLEDScreen:
    """
    OLED Screen Viewer - Real-time WebGL preview of how images look on OLED displays.

    This is a VIEW-ONLY node. It passes images through unchanged.
    The WebGL preview shows OLED simulation (pixel grid, color tint, glow).
    Use Koshi_Dither or Koshi_Greyscale nodes upstream for actual processing.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "view"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "screen_preset": (["SSD1363 256x128", "SSD1306 128x64", "SSD1306 128x32", "Custom"],),
                "custom_width": ("INT", {"default": 256, "min": 32, "max": 512, "step": 8}),
                "custom_height": ("INT", {"default": 128, "min": 32, "max": 256, "step": 8}),
                "resize_to_screen": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # These control the WebGL preview only (not image processing)
                "color_mode": (["grayscale", "green_mono", "blue_mono", "amber_mono", "white_mono", "yellow_mono"],),
                "show_pixel_grid": ("BOOLEAN", {"default": True}),
                "pixel_gap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "bloom_glow": ("BOOLEAN", {"default": False}),
                "bloom_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def view(
        self,
        images: torch.Tensor,
        screen_preset: str,
        custom_width: int,
        custom_height: int,
        resize_to_screen: bool,
        color_mode: str = "grayscale",
        show_pixel_grid: bool = True,
        pixel_gap: float = 0.15,
        bloom_glow: bool = False,
        bloom_intensity: float = 0.3,
    ) -> Tuple[torch.Tensor]:
        """
        Pass-through viewer. WebGL preview shows OLED simulation.
        Optionally resizes to screen dimensions.
        """
        # Get screen size from preset
        screen_sizes = {
            "SSD1363 256x128": (256, 128),
            "SSD1306 128x64": (128, 64),
            "SSD1306 128x32": (128, 32),
            "Custom": (custom_width, custom_height),
        }
        screen_w, screen_h = screen_sizes.get(screen_preset, (256, 128))

        if not resize_to_screen or not PIL_AVAILABLE:
            # Pass through unchanged with preview
            preview_images = save_images_for_preview(images)
            return {
                "ui": {"images": preview_images},
                "result": (images,)
            }

        # Resize to screen dimensions
        results = []
        for b in range(images.shape[0]):
            img_np = images[b].cpu().numpy()
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            pil_img = pil_img.resize((screen_w, screen_h), PILImage.LANCZOS)
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))

        output_tensor = torch.stack(results).to(images.device)
        preview_images = save_images_for_preview(output_tensor)
        return {
            "ui": {"images": preview_images},
            "result": (output_tensor,)
        }


NODE_CLASS_MAPPINGS = {
    "Koshi_SpriteSheet": KoshiSpriteSheet,
    "Koshi_OLEDScreen": KoshiOLEDScreen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_SpriteSheet": "░▒░ KN Sprite Sheet",
    "Koshi_OLEDScreen": "░▒░ KN OLED Screen",
}
