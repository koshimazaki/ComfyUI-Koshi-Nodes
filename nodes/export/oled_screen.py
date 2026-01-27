"""OLED Screen Emulation, Scaling, and Sprite Sheet Generation."""
import torch
import numpy as np
from typing import Tuple, List

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Resolution presets for common OLED displays
OLED_PRESETS = {
    "SSD1306 128x64": (128, 64),
    "SSD1306 128x32": (128, 32),
    "SSD1309 128x64": (128, 64),
    "SSD1322 256x64": (256, 64),
    "SSD1363 256x128": (256, 128),
    "SH1106 128x64": (128, 64),
    "Custom": (0, 0),
}

# Region presets for 256x128 display
REGION_PRESETS = {
    "full": (0, 0, 256, 128),           # Full screen
    "left_half": (0, 0, 128, 128),      # Left 128x128
    "right_half": (128, 0, 128, 128),   # Right 128x128
    "top_left_64": (0, 0, 64, 64),      # Top-left quadrant
    "top_right_64": (64, 0, 64, 64),
    "bottom_left_64": (0, 64, 64, 64),
    "bottom_right_64": (64, 64, 64, 64),
    "center_128": (64, 0, 128, 128),    # Centered 128x128
    "left_64": (0, 32, 64, 64),         # Left side centered vertically
    "right_64": (192, 32, 64, 64),      # Right side centered vertically
    "custom": (0, 0, 0, 0),
}

SCALE_METHODS = ["nearest", "lanczos", "bilinear", "bicubic", "box"]


class KoshiPixelScaler:
    """
    Scale images for pixel-perfect display on OLED screens.
    Supports Lanczos downscaling from generated images to target resolution.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "scale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scaled",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (list(OLED_PRESETS.keys()),),
                "custom_width": ("INT", {"default": 256, "min": 8, "max": 1024, "step": 8}),
                "custom_height": ("INT", {"default": 128, "min": 8, "max": 1024, "step": 8}),
                "method": (SCALE_METHODS, {"default": "lanczos"}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "fill_color": (["black", "white", "gray"],),
            }
        }

    def scale(
        self,
        images: torch.Tensor,
        preset: str,
        custom_width: int,
        custom_height: int,
        method: str,
        maintain_aspect: bool,
        fill_color: str,
    ) -> Tuple[torch.Tensor]:
        if not PIL_AVAILABLE:
            return (images,)

        # Get target size
        if preset == "Custom":
            target_w, target_h = custom_width, custom_height
        else:
            target_w, target_h = OLED_PRESETS[preset]

        # Map method to PIL
        pil_methods = {
            "nearest": PILImage.NEAREST,
            "lanczos": PILImage.LANCZOS,
            "bilinear": PILImage.BILINEAR,
            "bicubic": PILImage.BICUBIC,
            "box": PILImage.BOX,
        }
        resample = pil_methods.get(method, PILImage.LANCZOS)

        fill_values = {"black": 0, "white": 255, "gray": 128}
        fill = fill_values.get(fill_color, 0)

        results = []
        for b in range(images.shape[0]):
            img_np = (images[b].cpu().numpy() * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)

            if maintain_aspect:
                # Calculate aspect-preserving size
                src_w, src_h = pil_img.size
                scale = min(target_w / src_w, target_h / src_h)
                new_w = int(src_w * scale)
                new_h = int(src_h * scale)

                # Resize maintaining aspect
                resized = pil_img.resize((new_w, new_h), resample)

                # Create canvas and paste centered
                canvas = PILImage.new("RGB", (target_w, target_h), (fill, fill, fill))
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                canvas.paste(resized, (offset_x, offset_y))
                result = canvas
            else:
                result = pil_img.resize((target_w, target_h), resample)

            result_np = np.array(result).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))

        return (torch.stack(results).to(images.device),)


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
            # Pass through unchanged
            return (images,)

        # Resize to screen dimensions
        results = []
        for b in range(images.shape[0]):
            img_np = images[b].cpu().numpy()
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            pil_img = pil_img.resize((screen_w, screen_h), PILImage.LANCZOS)
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))

        return (torch.stack(results).to(images.device),)


class KoshiXBMExport:
    """
    Export images directly to XBM format for embedded displays.
    Supports variable names for direct C inclusion.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("xbm_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "sprite"}),
                "width": ("INT", {"default": 128, "min": 8, "max": 512, "step": 8}),
                "height": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
            }
        }

    def export(
        self,
        images: torch.Tensor,
        filename: str,
        width: int,
        height: int,
        threshold: float,
        invert: bool,
        output_path: str = "",
    ) -> Tuple[str]:
        import os

        if not PIL_AVAILABLE:
            return ("Error: PIL not available",)

        # Setup output directory
        if output_path and os.path.isdir(output_path):
            out_dir = output_path
        else:
            out_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "output", "xbm")
        os.makedirs(out_dir, exist_ok=True)

        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c == "_")
        if not safe_name:
            safe_name = "sprite"

        batch_size = images.shape[0]
        output_files = []

        for idx in range(batch_size):
            img_np = images[idx].cpu().numpy()

            # Resize to target
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            pil_img = pil_img.resize((width, height), PILImage.NEAREST)

            # Convert to grayscale and threshold
            gray = pil_img.convert("L")
            gray_np = np.array(gray).astype(np.float32) / 255.0
            binary = (gray_np > threshold).astype(np.uint8)

            if invert:
                binary = 1 - binary

            # Pack to bytes (MSB first, 8 pixels per byte)
            pad_w = (8 - width % 8) % 8
            if pad_w > 0:
                binary = np.pad(binary, ((0, 0), (0, pad_w)), constant_values=0)

            packed = np.zeros((height, binary.shape[1] // 8), dtype=np.uint8)
            for bit in range(8):
                packed |= (binary[:, bit::8] << bit)  # XBM uses LSB first!

            packed_bytes = packed.flatten()

            # Generate XBM content
            frame_name = f"{safe_name}_{idx:04d}" if batch_size > 1 else safe_name
            lines = [
                f"#define {frame_name}_width {width}",
                f"#define {frame_name}_height {height}",
                f"static unsigned char {frame_name}_bits[] = {{"
            ]

            hex_values = [f"0x{b:02x}" for b in packed_bytes]
            for i in range(0, len(hex_values), 12):
                lines.append("   " + ", ".join(hex_values[i:i+12]) + ",")

            lines.append("};")

            # Write file
            file_path = os.path.join(out_dir, f"{frame_name}.xbm")
            with open(file_path, "w") as f:
                f.write("\n".join(lines))

            output_files.append(file_path)

        return (output_files[0] if len(output_files) == 1 else out_dir,)


NODE_CLASS_MAPPINGS = {
    "Koshi_PixelScaler": KoshiPixelScaler,
    "Koshi_SpriteSheet": KoshiSpriteSheet,
    "Koshi_OLEDScreen": KoshiOLEDScreen,
    "Koshi_XBMExport": KoshiXBMExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_PixelScaler": "░▒░ KN Pixel Scaler",
    "Koshi_SpriteSheet": "░▒░ KN Sprite Sheet",
    "Koshi_OLEDScreen": "░▒░ KN OLED Screen",
    "Koshi_XBMExport": "░▒░ KN XBM Export",
}
