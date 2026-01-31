"""SIDKIT OLED Screen - Preview and Sprite Sheet for OLED displays."""

import torch
import numpy as np
import os
import uuid

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


def save_images_for_preview(image_tensor):
    """Save images to temp folder and return preview metadata."""
    if not COMFY_AVAILABLE or not PIL_AVAILABLE:
        return []

    results = []
    output_dir = folder_paths.get_temp_directory()

    batch = image_tensor if len(image_tensor.shape) == 4 else image_tensor.unsqueeze(0)

    for i in range(batch.shape[0]):
        img_np = (np.clip(batch[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)
        filename = f"sidkit_oled_{uuid.uuid4().hex[:8]}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)
        results.append({"filename": filename, "subfolder": "", "type": "temp"})

    return results


class SIDKITOLEDScreen:
    """
    SIDKIT OLED Screen Viewer - Real-time WebGL preview with video playback.
    Pass-through node that shows how images will look on OLED displays.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/SIDKIT"
    FUNCTION = "view"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    SCREEN_PRESETS = {
        "SSD1363 256x128": (256, 128),
        "SSD1306 128x64": (128, 64),
        "SSD1306 128x32": (128, 32),
        "SSD1322 256x64": (256, 64),
        "Custom": (0, 0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "screen_preset": (list(cls.SCREEN_PRESETS.keys()), {"default": "SSD1363 256x128"}),
                "resize_to_screen": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 256, "min": 32, "max": 512, "step": 8}),
                "custom_height": ("INT", {"default": 128, "min": 32, "max": 256, "step": 8}),
                "color_mode": (["grayscale", "green_mono", "blue_mono", "amber_mono",
                                "white_mono", "yellow_mono"], {"default": "grayscale"}),
                "show_pixel_grid": ("BOOLEAN", {"default": True}),
                "pixel_gap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "bloom_glow": ("BOOLEAN", {"default": False}),
                "bloom_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def view(self, images, screen_preset, resize_to_screen,
             custom_width=256, custom_height=128, color_mode="grayscale",
             show_pixel_grid=True, pixel_gap=0.15, bloom_glow=False, bloom_intensity=0.3):

        # Get screen dimensions
        if screen_preset == "Custom":
            screen_w, screen_h = custom_width, custom_height
        else:
            screen_w, screen_h = self.SCREEN_PRESETS.get(screen_preset, (256, 128))

        if not resize_to_screen or not PIL_AVAILABLE:
            preview_images = save_images_for_preview(images)
            return {"ui": {"images": preview_images}, "result": (images,)}

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

        return {"ui": {"images": preview_images}, "result": (output_tensor,)}


class SIDKITSpriteSheet:
    """
    Generate sprite sheet from frame sequence.
    Perfect for game dev exports and embedded display animations.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/SIDKIT"
    FUNCTION = "create_sheet"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("sprite_sheet", "cols", "rows", "frame_count")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "layout": (["auto", "horizontal", "vertical", "grid"], {"default": "auto"}),
                "max_cols": ("INT", {"default": 8, "min": 1, "max": 32}),
                "max_rows": ("INT", {"default": 8, "min": 1, "max": 32}),
            },
            "optional": {
                "padding": ("INT", {"default": 0, "min": 0, "max": 16}),
                "background": (["black", "white", "transparent", "magenta"], {"default": "black"}),
            }
        }

    def create_sheet(self, images, layout, max_cols, max_rows, padding=0, background="black"):
        batch_size = images.shape[0]
        h, w, c = images.shape[1], images.shape[2], images.shape[3]

        # Calculate grid dimensions
        if layout == "horizontal":
            cols, rows = batch_size, 1
        elif layout == "vertical":
            cols, rows = 1, batch_size
        elif layout == "grid":
            cols = min(max_cols, batch_size)
            rows = (batch_size + cols - 1) // cols
        else:  # auto
            cols = int(np.ceil(np.sqrt(batch_size)))
            rows = int(np.ceil(batch_size / cols))

        cols = min(cols, max_cols)
        rows = min(rows, max_rows)

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

        # Create sheet
        if background == "transparent":
            sheet = torch.zeros((sheet_h, sheet_w, 4), dtype=images.dtype, device=images.device)
        else:
            sheet = torch.zeros((sheet_h, sheet_w, c), dtype=images.dtype, device=images.device)
            for i, v in enumerate(bg[:c]):
                sheet[:, :, i] = v

        # Place frames
        for idx in range(min(batch_size, cols * rows)):
            row, col = idx // cols, idx % cols
            y, x = row * (h + padding), col * (w + padding)

            if background == "transparent":
                sheet[y:y+h, x:x+w, :c] = images[idx]
                sheet[y:y+h, x:x+w, 3] = 1.0
            else:
                sheet[y:y+h, x:x+w] = images[idx]

        output = sheet.unsqueeze(0)
        preview_images = save_images_for_preview(output)

        return {
            "ui": {"images": preview_images},
            "result": (output, cols, rows, batch_size)
        }


NODE_CLASS_MAPPINGS = {
    "SIDKIT_OLEDScreen": SIDKITOLEDScreen,
    "SIDKIT_SpriteSheet": SIDKITSpriteSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SIDKIT_OLEDScreen": "░▒░ SIDKIT OLED Screen",
    "SIDKIT_SpriteSheet": "░▒░ SIDKIT Sprite Sheet",
}
