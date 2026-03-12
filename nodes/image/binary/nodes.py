"""KN SIDKIT Binary — Threshold, OLED screen emulation preview, and SIDV/C export."""
import torch
import numpy as np
import struct
from typing import Tuple

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import folder_paths
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False


def save_images_for_preview(image_tensor, prefix="koshi_sidkit"):
    """Save images to temp folder for ComfyUI preview."""
    if not PREVIEW_AVAILABLE:
        return []
    import os, uuid
    results = []
    output_dir = folder_paths.get_temp_directory()
    batch = image_tensor if len(image_tensor.shape) == 4 else image_tensor.unsqueeze(0)
    for i in range(batch.shape[0]):
        import numpy as np
        img_np = (np.clip(batch[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{i}.png"
        PILImage.fromarray(img_np).save(os.path.join(output_dir, filename))
        results.append({"filename": filename, "subfolder": "", "type": "temp"})
    return results


# SIDV format: SIDK magic + header fields
SIDV_MAGIC = b'SIDK'
SIDV_VERSION = 1


class KoshiSIDKITBinary:
    """
    SIDKIT Binary converter with inline OLED screen emulation.
    Converts images to 1/2/4-bit binary with dithering.
    Exports to SIDV binary format and C headers for Teensy.
    WebGL OLED preview with bloom, pixel grid, and colour tint.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/SIDKIT"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("binary", "hex_data", "sidv_path")
    OUTPUT_NODE = True

    METHODS = ["simple", "adaptive", "otsu", "dither_bayer", "dither_floyd", "dither_atkinson"]
    BIT_DEPTHS = ["1-bit (mono)", "2-bit (4 levels)", "4-bit (16 levels)"]
    EXPORT_FORMATS = ["none", "sidv", "xbm", "c_header", "sidv + c_header"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (cls.METHODS, {"default": "dither_atkinson"}),
                "bit_depth": (cls.BIT_DEPTHS, {"default": "1-bit (mono)"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "screen_preset": (["SSD1363 256x128", "SSD1306 128x64", "SSD1306 128x32", "Custom"],
                                  {"default": "SSD1363 256x128"}),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 256, "min": 32, "max": 512, "step": 8}),
                "custom_height": ("INT", {"default": 128, "min": 32, "max": 256, "step": 8}),
                "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                "adaptive_c": ("FLOAT", {"default": 2.0, "min": -20.0, "max": 20.0, "step": 0.5}),
                "export_format": (cls.EXPORT_FORMATS, {"default": "sidv"}),
                "export_name": ("STRING", {"default": "sidkit_export"}),
                "fps": ("INT", {"default": 60, "min": 1, "max": 120}),
                # Preview controls (WebGL only)
                "color_mode": (["grayscale", "green_mono", "blue_mono", "amber_mono", "white_mono", "yellow_mono"],
                               {"default": "yellow_mono"}),
                "show_pixel_grid": ("BOOLEAN", {"default": True}),
                "pixel_gap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "bloom_glow": ("BOOLEAN", {"default": True}),
                "bloom_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def _get_screen_size(self, screen_preset, custom_width, custom_height):
        sizes = {
            "SSD1363 256x128": (256, 128),
            "SSD1306 128x64": (128, 64),
            "SSD1306 128x32": (128, 32),
            "Custom": (custom_width, custom_height),
        }
        return sizes.get(screen_preset, (256, 128))

    def _parse_bit_depth(self, bit_depth_str):
        if "1-bit" in bit_depth_str:
            return 1
        elif "2-bit" in bit_depth_str:
            return 2
        return 4

    def _to_gray(self, img):
        if len(img.shape) == 3 and img.shape[2] >= 3:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img[:, :, 0] if len(img.shape) == 3 else img

    def _local_mean(self, gray, block_size):
        try:
            from scipy.ndimage import uniform_filter
            return uniform_filter(gray.astype(np.float64), size=block_size, mode='reflect')
        except ImportError:
            pad = block_size // 2
            padded = np.pad(gray, pad, mode='reflect')
            cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            h, w = gray.shape
            result = np.zeros_like(gray, dtype=np.float64)
            for y in range(h):
                for x in range(w):
                    y1, y2 = y, y + block_size
                    x1, x2 = x, x + block_size
                    result[y, x] = (cumsum[y2, x2] - cumsum[y1, x2] - cumsum[y2, x1] + cumsum[y1, x1]) / (block_size ** 2)
            return result

    def _otsu_threshold(self, gray):
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

    def _bayer_matrix(self, n):
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        smaller = self._bayer_matrix(n // 2)
        return np.block([[4*smaller, 4*smaller+2], [4*smaller+3, 4*smaller+1]]) / (n*n)

    def _atkinson_dither(self, gray, levels):
        img = gray.copy()
        h, w = img.shape
        step = 1.0 / max(levels - 1, 1)
        for y in range(h):
            for x in range(w):
                old = img[y, x]
                new = np.round(old / step) * step
                new = np.clip(new, 0.0, 1.0)
                img[y, x] = new
                err = (old - new) / 8.0
                if x + 1 < w: img[y, x+1] += err
                if x + 2 < w: img[y, x+2] += err
                if y + 1 < h:
                    if x > 0: img[y+1, x-1] += err
                    img[y+1, x] += err
                    if x + 1 < w: img[y+1, x+1] += err
                if y + 2 < h: img[y+2, x] += err
        return np.clip(img, 0, 1)

    def _quantize(self, value, levels):
        step = 1.0 / max(levels - 1, 1)
        return np.round(value / step) * step

    def _apply_method(self, gray, method, threshold, bit_depth, block_size, adaptive_c):
        levels = 2 ** bit_depth

        if method == "simple":
            if bit_depth == 1:
                return (gray > threshold).astype(np.float32)
            return self._quantize(gray, levels)

        elif method == "adaptive":
            local_mean = self._local_mean(gray, block_size)
            if bit_depth == 1:
                return (gray > (local_mean - adaptive_c / 255.0)).astype(np.float32)
            diff = gray - local_mean + adaptive_c / 255.0
            return self._quantize(np.clip(diff + 0.5, 0, 1), levels)

        elif method == "otsu":
            auto_thresh = self._otsu_threshold(gray)
            if bit_depth == 1:
                return (gray > auto_thresh).astype(np.float32)
            return self._quantize(gray, levels)

        elif method == "dither_bayer":
            bayer = self._bayer_matrix(8 if bit_depth > 1 else 4)
            h, w = gray.shape
            bs = bayer.shape[0]
            tiled = np.tile(bayer, ((h + bs - 1) // bs, (w + bs - 1) // bs))[:h, :w]
            if bit_depth == 1:
                return (gray > tiled).astype(np.float32)
            dithered = gray + (tiled - 0.5) * (1.0 / levels)
            return np.clip(self._quantize(dithered, levels), 0, 1)

        elif method == "dither_floyd":
            result = gray.copy()
            h, w = gray.shape
            step = 1.0 / max(levels - 1, 1)
            for y in range(h):
                for x in range(w):
                    old = result[y, x]
                    new = np.round(old / step) * step
                    new = np.clip(new, 0.0, 1.0)
                    result[y, x] = new
                    err = old - new
                    if x + 1 < w: result[y, x+1] += err * 7/16
                    if y + 1 < h:
                        if x > 0: result[y+1, x-1] += err * 3/16
                        result[y+1, x] += err * 5/16
                        if x + 1 < w: result[y+1, x+1] += err * 1/16
            return np.clip(result, 0, 1)

        elif method == "dither_atkinson":
            return self._atkinson_dither(gray, levels)

        return (gray > threshold).astype(np.float32)

    def _pack_1bit(self, binary):
        h, w = binary.shape
        pad_w = (8 - w % 8) % 8
        if pad_w > 0:
            binary = np.pad(binary, ((0, 0), (0, pad_w)), constant_values=0)
        packed = np.zeros((h, binary.shape[1] // 8), dtype=np.uint8)
        for bit in range(8):
            packed |= (binary[:, bit::8].astype(np.uint8) << (7 - bit))
        return packed

    def _pack_2bit(self, img):
        quantized = np.clip(np.round(img * 3), 0, 3).astype(np.uint8)
        h, w = quantized.shape
        pad_w = (4 - w % 4) % 4
        if pad_w > 0:
            quantized = np.pad(quantized, ((0, 0), (0, pad_w)), constant_values=0)
        packed = np.zeros((h, quantized.shape[1] // 4), dtype=np.uint8)
        for i in range(4):
            packed |= (quantized[:, i::4] << (6 - i * 2))
        return packed

    def _pack_4bit(self, img):
        quantized = np.clip(np.round(img * 15), 0, 15).astype(np.uint8)
        h, w = quantized.shape
        pad_w = (2 - w % 2) % 2
        if pad_w > 0:
            quantized = np.pad(quantized, ((0, 0), (0, pad_w)), constant_values=0)
        packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]
        return packed

    def _generate_c_header(self, frames, width, height, bit_depth, name, fps):
        mono = bit_depth == 1
        if bit_depth == 1:
            bpf = (width * height) // 8
        elif bit_depth == 2:
            bpf = (width * height) // 4
        else:
            bpf = (width * height) // 2
        total_bytes = bpf * len(frames)

        lines = [
            "// SIDKIT OLED Converter Output",
            f"// Target: SSD1363 {width}x{height} {bit_depth}-bit",
            f"// Frames: {len(frames)} @ {fps} FPS = {len(frames)/fps:.1f}s",
            f"// Memory: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)",
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"#define {name.upper()}_WIDTH {width}",
            f"#define {name.upper()}_HEIGHT {height}",
            f"#define {name.upper()}_BIT_DEPTH {bit_depth}",
            f"#define {name.upper()}_FRAME_COUNT {len(frames)}",
            f"#define {name.upper()}_FPS {fps}",
            f"#define {name.upper()}_FRAME_DELAY_MS {1000 // fps}",
            f"#define {name.upper()}_BYTES_PER_FRAME {bpf}",
            "",
        ]

        for i, frame in enumerate(frames):
            if bit_depth == 1:
                packed = self._pack_1bit((frame > 0.5).astype(np.uint8))
            elif bit_depth == 2:
                packed = self._pack_2bit(frame)
            else:
                packed = self._pack_4bit(frame)
            flat = packed.flatten()
            lines.append(f"const uint8_t PROGMEM {name}_frame_{i}[{len(flat)}] = {{")
            for row_start in range(0, len(flat), 16):
                row = flat[row_start:row_start + 16]
                lines.append("    " + ", ".join(f"0x{b:02X}" for b in row) + ",")
            lines.append("};")
            lines.append("")

        lines.append(f"const uint8_t* const {name}_frames[{len(frames)}] = {{")
        for i in range(len(frames)):
            lines.append(f"    {name}_frame_{i},")
        lines.append("};")

        return "\n".join(lines)

    def _generate_xbm(self, frames, width, height, name):
        """Generate XBM format (1-bit only, LSB first)."""
        lines = [
            f"#define {name}_width {width}",
            f"#define {name}_height {height}",
        ]
        for i, frame in enumerate(frames):
            binary = (frame > 0.5).astype(np.uint8)
            h, w = binary.shape
            pad_w = (8 - w % 8) % 8
            if pad_w > 0:
                binary = np.pad(binary, ((0, 0), (0, pad_w)), constant_values=0)
            packed = np.zeros((h, binary.shape[1] // 8), dtype=np.uint8)
            for bit in range(8):
                packed |= (binary[:, bit::8] << bit)  # LSB first for XBM
            flat = packed.flatten()
            suffix = f"_{i}" if len(frames) > 1 else ""
            lines.append(f"static unsigned char {name}{suffix}_bits[] = {{")
            for row_start in range(0, len(flat), 12):
                row = flat[row_start:row_start + 12]
                lines.append("   " + ", ".join(f"0x{b:02x}" for b in row) + ",")
            lines.append("};")
            lines.append("")
        return "\n".join(lines)

    def _generate_sidv(self, frames, width, height, bit_depth, fps):
        """Generate SIDV binary: SIDK header + packed frame data."""
        # Header: SIDK(4) + version(1) + width(1) + height(1) + bpp(1) + fps(1) + frame_count(2) + reserved(6) = 16 bytes
        frame_count = len(frames)
        header = SIDV_MAGIC
        header += struct.pack('>B', SIDV_VERSION)
        header += struct.pack('>B', width & 0xFF)  # width low byte (high bit in next)
        header += struct.pack('>B', (width >> 8) & 0xFF)
        header += struct.pack('>B', height & 0xFF)
        header += struct.pack('>B', bit_depth)
        header += struct.pack('>B', fps)
        header += struct.pack('>H', frame_count)
        header += b'\x00' * 4  # reserved

        # Pack frames
        frame_data = bytearray()
        for frame in frames:
            if bit_depth == 1:
                packed = self._pack_1bit((frame > 0.5).astype(np.uint8))
            elif bit_depth == 2:
                packed = self._pack_2bit(frame)
            else:
                packed = self._pack_4bit(frame)
            frame_data.extend(packed.flatten().tobytes())

        return bytes(header) + bytes(frame_data)

    def convert(
        self,
        image,
        method,
        bit_depth,
        threshold,
        invert,
        screen_preset,
        show_preview,
        custom_width=256,
        custom_height=128,
        block_size=11,
        adaptive_c=2.0,
        export_format="sidv",
        export_name="sidkit_export",
        fps=60,
        color_mode="yellow_mono",
        show_pixel_grid=True,
        pixel_gap=0.15,
        bloom_glow=True,
        bloom_intensity=0.3,
    ):
        screen_w, screen_h = self._get_screen_size(screen_preset, custom_width, custom_height)
        bpp = self._parse_bit_depth(bit_depth)

        results = []
        gray_frames = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()
            gray = self._to_gray(img_np)

            # Resize to screen dimensions
            if PIL_AVAILABLE and (gray.shape[1] != screen_w or gray.shape[0] != screen_h):
                pil_img = PILImage.fromarray((gray * 255).astype(np.uint8))
                pil_img = pil_img.resize((screen_w, screen_h), PILImage.LANCZOS)
                gray = np.array(pil_img).astype(np.float32) / 255.0

            # Apply binary conversion
            binary = self._apply_method(gray, method, threshold, bpp, block_size, adaptive_c)

            if invert:
                binary = 1.0 - binary

            gray_frames.append(binary)

            # Output as RGB for ComfyUI preview
            result = np.stack([binary, binary, binary], axis=-1)
            results.append(torch.from_numpy(result.astype(np.float32)))

        output_tensor = torch.stack(results).to(image.device)

        # Export based on selected format — all files go to ComfyUI output dir
        import os
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        hex_str = ""
        export_path = ""

        if export_format in ("c_header", "sidv + c_header"):
            hex_str = self._generate_c_header(gray_frames, screen_w, screen_h, bpp, export_name, fps)
            c_filename = f"{export_name}_{screen_w}x{screen_h}_{bpp}bit.h"
            with open(os.path.join(output_dir, c_filename), 'w') as f:
                f.write(hex_str)
            export_path = c_filename

        if export_format == "xbm":
            hex_str = self._generate_xbm(gray_frames, screen_w, screen_h, export_name)
            xbm_filename = f"{export_name}_{screen_w}x{screen_h}.xbm"
            with open(os.path.join(output_dir, xbm_filename), 'w') as f:
                f.write(hex_str)
            export_path = xbm_filename

        if export_format in ("sidv", "sidv + c_header"):
            sidv_data = self._generate_sidv(gray_frames, screen_w, screen_h, bpp, fps)
            sidv_filename = f"{export_name}_{screen_w}x{screen_h}_{bpp}bit_{fps}fps.sidv"
            with open(os.path.join(output_dir, sidv_filename), 'wb') as f:
                f.write(sidv_data)
            export_path = sidv_filename

        # Return with oled_frames (custom key) so ComfyUI doesn't add default preview
        # The WebGL OLED preview JS picks these up via onExecuted
        preview_images = save_images_for_preview(output_tensor)
        return {
            "ui": {"oled_frames": preview_images},
            "result": (output_tensor, hex_str, export_path)
        }


# Keep old class for backwards compatibility
KoshiBinary = KoshiSIDKITBinary

NODE_CLASS_MAPPINGS = {
    "Koshi_Binary": KoshiSIDKITBinary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Binary": "░▒░ KN SIDKIT Binary",
}
