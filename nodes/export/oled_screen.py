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
    Enhanced OLED screen emulator with multiple display modes.
    Simulates real hardware characteristics including pixel gaps, color tints, and burn-in.
    Supports region presets for placing content in specific screen areas.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "emulate"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "raw_screen")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "screen_preset": (["SSD1363 256x128", "SSD1306 128x64", "Custom"],),
                "region": (list(REGION_PRESETS.keys()),),
                "custom_x": ("INT", {"default": 0, "min": 0, "max": 256, "step": 8}),
                "custom_y": ("INT", {"default": 0, "min": 0, "max": 128, "step": 8}),
                "custom_w": ("INT", {"default": 128, "min": 8, "max": 256, "step": 8}),
                "custom_h": ("INT", {"default": 128, "min": 8, "max": 128, "step": 8}),
                "color_mode": (["grayscale", "green_mono", "blue_mono", "amber_mono", "rgb"],),
                "bit_depth": (["1-bit", "2-bit", "4-bit", "8-bit"],),
                "dither_type": (["none", "bayer_2x2", "bayer_4x4", "bayer_8x8", "floyd_steinberg"],),
                "preview_scale": ("INT", {"default": 4, "min": 1, "max": 8}),
                "show_pixel_grid": ("BOOLEAN", {"default": True}),
                "pixel_gap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
            },
            "optional": {
                "bezel": ("BOOLEAN", {"default": False}),
                "bezel_color": (["black", "silver", "gold"],),
            }
        }

    def _bayer_matrix(self, n: int) -> np.ndarray:
        """Generate Bayer dither matrix of size n x n."""
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ]) / (n * n)

    def _floyd_steinberg_dither(self, img: np.ndarray, levels: int) -> np.ndarray:
        """Apply Floyd-Steinberg error diffusion dithering."""
        h, w = img.shape
        result = img.astype(np.float32).copy()

        for y in range(h):
            for x in range(w):
                old_val = result[y, x]
                new_val = np.round(old_val * (levels - 1)) / (levels - 1)
                result[y, x] = new_val
                error = old_val - new_val

                if x + 1 < w:
                    result[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        result[y + 1, x - 1] += error * 3 / 16
                    result[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        result[y + 1, x + 1] += error * 1 / 16

        return np.clip(result, 0, 1)

    def _apply_dither(self, gray: np.ndarray, dither_type: str, levels: int) -> np.ndarray:
        """Apply dithering to grayscale image."""
        if dither_type == "none":
            return np.floor(gray * (levels - 1) + 0.5) / (levels - 1)
        elif dither_type == "floyd_steinberg":
            return self._floyd_steinberg_dither(gray, levels)
        else:
            # Bayer dithering
            size = int(dither_type.split("_")[1].split("x")[0])
            bayer = self._bayer_matrix(size)
            h, w = gray.shape
            tile_y = (h + size - 1) // size
            tile_x = (w + size - 1) // size
            threshold_map = np.tile(bayer, (tile_y, tile_x))[:h, :w]
            dithered = gray + (threshold_map - 0.5) / levels
            return np.clip(np.floor(dithered * (levels - 1) + 0.5) / (levels - 1), 0, 1)

    def _apply_color_mode(self, gray: np.ndarray, color_mode: str) -> np.ndarray:
        """Apply color tint based on display type."""
        h, w = gray.shape

        if color_mode == "grayscale":
            # True OLED: slight blue tint on whites
            r = gray * 0.95
            g = gray * 1.0
            b = gray * 1.02
        elif color_mode == "green_mono":
            r = gray * 0.1
            g = gray * 1.0
            b = gray * 0.1
        elif color_mode == "blue_mono":
            r = gray * 0.2
            g = gray * 0.4
            b = gray * 1.0
        elif color_mode == "amber_mono":
            r = gray * 1.0
            g = gray * 0.6
            b = gray * 0.1
        else:  # rgb - pass through
            return np.stack([gray, gray, gray], axis=-1)

        return np.clip(np.stack([r, g, b], axis=-1), 0, 1)

    def emulate(
        self,
        images: torch.Tensor,
        screen_preset: str,
        region: str,
        custom_x: int,
        custom_y: int,
        custom_w: int,
        custom_h: int,
        color_mode: str,
        bit_depth: str,
        dither_type: str,
        preview_scale: int,
        show_pixel_grid: bool,
        pixel_gap: float,
        bezel: bool = False,
        bezel_color: str = "black",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not PIL_AVAILABLE:
            return (images, images)

        # Get screen size from preset
        screen_sizes = {
            "SSD1363 256x128": (256, 128),
            "SSD1306 128x64": (128, 64),
            "Custom": (256, 128),
        }
        screen_w, screen_h = screen_sizes.get(screen_preset, (256, 128))

        # Get region (where to place content)
        if region == "custom":
            reg_x, reg_y, reg_w, reg_h = custom_x, custom_y, custom_w, custom_h
        else:
            reg_x, reg_y, reg_w, reg_h = REGION_PRESETS.get(region, (0, 0, screen_w, screen_h))
            # Scale region if screen is not 256x128
            if screen_w != 256 or screen_h != 128:
                scale_x, scale_y = screen_w / 256, screen_h / 128
                reg_x = int(reg_x * scale_x)
                reg_y = int(reg_y * scale_y)
                reg_w = int(reg_w * scale_x)
                reg_h = int(reg_h * scale_y)

        # Clamp region to screen bounds
        reg_w = min(reg_w, screen_w - reg_x)
        reg_h = min(reg_h, screen_h - reg_y)

        # Parse bit depth
        levels = {"1-bit": 2, "2-bit": 4, "4-bit": 16, "8-bit": 256}[bit_depth]

        previews = []
        raw_screens = []

        for b in range(images.shape[0]):
            img_np = images[b].cpu().numpy()

            # Resize input to region size
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            pil_img = pil_img.resize((reg_w, reg_h), PILImage.LANCZOS)
            img_resized = np.array(pil_img).astype(np.float32) / 255.0

            # Convert to grayscale (for non-RGB modes)
            if color_mode != "rgb":
                if len(img_resized.shape) == 3:
                    gray = 0.299 * img_resized[:, :, 0] + 0.587 * img_resized[:, :, 1] + 0.114 * img_resized[:, :, 2]
                else:
                    gray = img_resized

                # Apply dithering
                gray = self._apply_dither(gray, dither_type, levels)
                region_data = gray
            else:
                # RGB mode - quantize each channel
                region_data = np.zeros((reg_h, reg_w), dtype=np.float32)
                for c in range(3):
                    region_data += self._apply_dither(img_resized[:, :, c], dither_type, levels) / 3

            # Create full screen canvas (black background for OLED)
            screen_gray = np.zeros((screen_h, screen_w), dtype=np.float32)

            # Place region content on canvas
            screen_gray[reg_y:reg_y + reg_h, reg_x:reg_x + reg_w] = region_data

            # Raw screen output (grayscale, ready for export)
            raw_screen = np.stack([screen_gray, screen_gray, screen_gray], axis=-1)
            raw_screens.append(torch.from_numpy(raw_screen.astype(np.float32)))

            # Apply color mode for preview
            colored = self._apply_color_mode(screen_gray, color_mode)

            # Scale up for preview
            output_h = screen_h * preview_scale
            output_w = screen_w * preview_scale

            if show_pixel_grid and preview_scale >= 2:
                # Render with pixel gaps
                gap = max(1, int(pixel_gap * preview_scale))
                output = np.zeros((output_h, output_w, 3), dtype=np.float32)
                output[:] = 0.01  # OLED black

                for y in range(screen_h):
                    for x in range(screen_w):
                        pixel_color = colored[y, x]
                        y1, y2 = y * preview_scale, (y + 1) * preview_scale - gap
                        x1, x2 = x * preview_scale, (x + 1) * preview_scale - gap

                        if y2 > y1 and x2 > x1:
                            output[y1:y2, x1:x2] = pixel_color
            else:
                pil_preview = PILImage.fromarray((colored * 255).astype(np.uint8))
                pil_preview = pil_preview.resize((output_w, output_h), PILImage.NEAREST)
                output = np.array(pil_preview).astype(np.float32) / 255.0

            # Add bezel if requested
            if bezel:
                bezel_width = max(preview_scale * 2, 4)
                bezel_colors = {
                    "black": [0.05, 0.05, 0.05],
                    "silver": [0.6, 0.6, 0.65],
                    "gold": [0.7, 0.55, 0.3],
                }
                bc = bezel_colors.get(bezel_color, [0.05, 0.05, 0.05])

                bezeled = np.zeros((output_h + bezel_width * 2, output_w + bezel_width * 2, 3), dtype=np.float32)
                bezeled[:] = bc
                bezeled[bezel_width:-bezel_width, bezel_width:-bezel_width] = output
                output = bezeled

            previews.append(torch.from_numpy(output))

        return (
            torch.stack(previews).to(images.device),
            torch.stack(raw_screens).to(images.device),
        )


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
    "Koshi_PixelScaler": "Koshi Pixel Scaler",
    "Koshi_SpriteSheet": "Koshi Sprite Sheet",
    "Koshi_OLEDScreen": "Koshi OLED Screen",
    "Koshi_XBMExport": "Koshi XBM Export",
}
