"""OLED Screen Preview Node - Preview how export will look on hardware."""
import torch
import numpy as np
import os
import uuid
from PIL import Image as PILImage

# ComfyUI imports for preview
try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


def save_images_for_preview(image_tensor):
    """Save images to temp folder and return preview metadata."""
    if not COMFY_AVAILABLE:
        return []

    results = []
    output_dir = folder_paths.get_temp_directory()

    # Handle batch
    if len(image_tensor.shape) == 4:
        batch = image_tensor
    else:
        batch = image_tensor.unsqueeze(0)

    for i in range(batch.shape[0]):
        img_np = batch[i].cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        pil_img = PILImage.fromarray(img_np)
        filename = f"koshi_oled_{uuid.uuid4().hex[:8]}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)

        results.append({
            "filename": filename,
            "subfolder": "",
            "type": "temp"
        })

    return results


class KoshiOLEDPreview:
    """
    Preview how images will look on OLED displays.
    Emulates SSD1363 256x128 4-bit greyscale display.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/Export"
    FUNCTION = "preview"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "screen_width": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 512,
                    "step": 8
                }),
                "screen_height": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 256,
                    "step": 8
                }),
                "bit_depth": (["1-bit (mono)", "2-bit (4 levels)", "4-bit (16 levels)"], {
                    "default": "4-bit (16 levels)"
                }),
                "dither": ("BOOLEAN", {"default": True}),
                "show_pixel_grid": ("BOOLEAN", {"default": True}),
                "scale": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
            }
        }

    def _parse_bit_depth(self, bit_depth_str):
        if "1-bit" in bit_depth_str:
            return 1
        elif "2-bit" in bit_depth_str:
            return 2
        elif "4-bit" in bit_depth_str:
            return 4
        return 4

    def _bayer_matrix(self, n):
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        else:
            smaller = self._bayer_matrix(n // 2)
            return np.block([
                [4 * smaller,     4 * smaller + 2],
                [4 * smaller + 3, 4 * smaller + 1]
            ]) / (n * n)

    def preview(self, image, screen_width, screen_height, bit_depth, dither, show_pixel_grid, scale):
        batch_size = image.shape[0]
        results = []
        
        bit_depth_int = self._parse_bit_depth(bit_depth)
        levels = 2 ** bit_depth_int
        
        # Bayer matrix for dithering
        bayer = self._bayer_matrix(8)
        
        for b in range(batch_size):
            img_np = image[b].cpu().numpy()
            
            # Resize to screen size (nearest neighbor for pixel art look)
            from PIL import Image as PILImage
            h, w = img_np.shape[:2]
            pil_img = PILImage.fromarray((img_np * 255).astype(np.uint8))
            pil_img = pil_img.resize((screen_width, screen_height), PILImage.NEAREST)
            img_resized = np.array(pil_img).astype(np.float32) / 255.0
            
            # Convert to greyscale
            if len(img_resized.shape) == 3:
                grey = 0.299 * img_resized[:,:,0] + 0.587 * img_resized[:,:,1] + 0.114 * img_resized[:,:,2]
            else:
                grey = img_resized
            
            # Apply dithering
            if dither:
                tile_y = (screen_height + 7) // 8
                tile_x = (screen_width + 7) // 8
                threshold_map = np.tile(bayer, (tile_y, tile_x))[:screen_height, :screen_width]
                grey = grey + (threshold_map - 0.5) / levels
            
            # Quantize to bit depth
            grey = np.clip(grey, 0, 1)
            quantized = np.floor(grey * (levels - 1) + 0.5) / (levels - 1)
            
            # Scale up for preview
            output_h = screen_height * scale
            output_w = screen_width * scale
            
            # Create output with OLED-like appearance
            output = np.zeros((output_h, output_w, 3), dtype=np.float32)
            
            for y in range(screen_height):
                for x in range(screen_width):
                    val = quantized[y, x]
                    
                    # OLED color (slight blue tint on brights, true black on dark)
                    if val < 0.01:
                        color = np.array([0.0, 0.0, 0.0])
                    else:
                        color = np.array([val * 0.95, val, val * 1.02])
                    
                    # Fill scaled pixel
                    y1, y2 = y * scale, (y + 1) * scale
                    x1, x2 = x * scale, (x + 1) * scale
                    
                    if show_pixel_grid and scale >= 2:
                        # Pixel with gap
                        gap = max(1, scale // 8)
                        output[y1:y2-gap, x1:x2-gap] = color
                        # Dark gap
                        output[y2-gap:y2, x1:x2] = 0.02
                        output[y1:y2, x2-gap:x2] = 0.02
                    else:
                        output[y1:y2, x1:x2] = color
            
            output = np.clip(output, 0, 1)
            results.append(output)

        output_tensor = torch.from_numpy(np.stack(results))

        # Return with preview UI
        preview_images = save_images_for_preview(output_tensor)
        return {
            "ui": {"images": preview_images},
            "result": (output_tensor,)
        }


NODE_CLASS_MAPPINGS = {
    "Koshi_OLEDPreview": KoshiOLEDPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_OLEDPreview": "Koshi OLED Preview",
}
