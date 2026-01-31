"""
SIDKIT Export Node - Unified export to all SIDKIT formats.
Outputs: .sidv (Teensy video), .xbm (standard), .h (C header), .bin (raw)

.sidv format:
  Header (16 bytes):
    0-3:  "SIDK" magic
    4:    version (1)
    5-6:  width (LE)
    7-8:  height (LE)
    9:    bitDepth (1/2/4)
    10:   fps
    11-14: frameCount (LE)
    15:   reserved
  Data: Packed frames
"""

import torch
import numpy as np
import struct
import os
from datetime import datetime


class SIDKITExport:
    """
    Export images/video to SIDKIT-compatible formats for Teensy/OLED displays.
    Combines all export functionality: .sidv, .xbm, .h, .bin
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    CATEGORY = "Koshi/SIDKIT"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("file_path", "hex_data", "frame_count", "file_size")
    OUTPUT_NODE = True

    OLED_PRESETS = {
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
                "format": (["sidv", "xbm", "c_header", "binary"], {"default": "sidv"}),
                "bit_depth": (["1-bit (mono)", "2-bit (4 levels)", "4-bit (16 levels)"],
                              {"default": "1-bit (mono)"}),
                "screen_preset": (list(cls.OLED_PRESETS.keys()), {"default": "SSD1363 256x128"}),
                "filename": ("STRING", {"default": "sidkit_export"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 256, "min": 32, "max": 512, "step": 8}),
                "custom_height": ("INT", {"default": 128, "min": 32, "max": 256, "step": 8}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": ""}),
                "show_hex": ("BOOLEAN", {"default": False}),
            }
        }

    def _get_dimensions(self, preset, custom_w, custom_h):
        if preset == "Custom":
            return custom_w, custom_h
        return self.OLED_PRESETS.get(preset, (256, 128))

    def _parse_bit_depth(self, bit_depth_str):
        if "1-bit" in bit_depth_str:
            return 1
        elif "2-bit" in bit_depth_str:
            return 2
        elif "4-bit" in bit_depth_str:
            return 4
        return 1

    def _resize_frame(self, frame, target_w, target_h):
        h, w = frame.shape[:2]
        if h == target_h and w == target_w:
            return frame
        y_ratio, x_ratio = h / target_h, w / target_w
        y_idx = np.clip((np.arange(target_h) * y_ratio).astype(int), 0, h - 1)
        x_idx = np.clip((np.arange(target_w) * x_ratio).astype(int), 0, w - 1)
        return frame[y_idx][:, x_idx]

    def _to_grayscale(self, frame):
        if len(frame.shape) == 2:
            return frame
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

    def _quantize(self, gray, bit_depth, threshold=0.5):
        if bit_depth == 1:
            return (gray > threshold).astype(np.uint8)
        elif bit_depth == 2:
            return np.clip((gray * 4).astype(np.uint8), 0, 3)
        elif bit_depth == 4:
            return np.clip((gray * 16).astype(np.uint8), 0, 15)
        return (gray > threshold).astype(np.uint8)

    def _pack_1bit(self, frame):
        h, w = frame.shape
        pad_w = (8 - w % 8) % 8
        if pad_w > 0:
            frame = np.pad(frame, ((0, 0), (0, pad_w)), constant_values=0)
        packed = np.zeros((h, frame.shape[1] // 8), dtype=np.uint8)
        for bit in range(8):
            packed |= (frame[:, bit::8] << (7 - bit))
        return packed.flatten()

    def _pack_2bit(self, frame):
        h, w = frame.shape
        pad_w = (4 - w % 4) % 4
        if pad_w > 0:
            frame = np.pad(frame, ((0, 0), (0, pad_w)), constant_values=0)
        packed = np.zeros((h, frame.shape[1] // 4), dtype=np.uint8)
        for i in range(4):
            packed |= (frame[:, i::4] & 0x03) << (6 - i * 2)
        return packed.flatten()

    def _pack_4bit(self, frame):
        h, w = frame.shape
        if w % 2:
            frame = np.pad(frame, ((0, 0), (0, 1)), constant_values=0)
        packed = (frame[:, 0::2] << 4) | (frame[:, 1::2] & 0x0F)
        return packed.flatten()

    def _pack_frame(self, frame, bit_depth):
        if bit_depth == 1:
            return self._pack_1bit(frame)
        elif bit_depth == 2:
            return self._pack_2bit(frame)
        elif bit_depth == 4:
            return self._pack_4bit(frame)
        return self._pack_1bit(frame)

    def _write_sidv(self, frames, w, h, bit_depth, fps, filepath):
        header = bytearray(16)
        header[0:4] = b'SIDK'
        header[4] = 1
        struct.pack_into('<H', header, 5, w)
        struct.pack_into('<H', header, 7, h)
        header[9] = bit_depth
        header[10] = fps
        struct.pack_into('<I', header, 11, len(frames))
        with open(filepath, 'wb') as f:
            f.write(header)
            for frame in frames:
                f.write(frame.tobytes())
        return os.path.getsize(filepath)

    def _write_xbm(self, frames, w, h, filename, filepath):
        if len(frames) == 1:
            lines = [
                f"#define {filename}_width {w}",
                f"#define {filename}_height {h}",
                f"static unsigned char {filename}_bits[] = {{"
            ]
            hex_vals = [f"0x{b:02x}" for b in frames[0]]
            for i in range(0, len(hex_vals), 12):
                lines.append("   " + ", ".join(hex_vals[i:i+12]) + ",")
            lines.append("};")
            with open(filepath, 'w') as f:
                f.write("\n".join(lines))
        else:
            base, ext = os.path.splitext(filepath)
            for i, frame in enumerate(frames):
                frame_path = f"{base}_{i:04d}{ext}"
                frame_name = f"{filename}_{i:04d}"
                lines = [
                    f"#define {frame_name}_width {w}",
                    f"#define {frame_name}_height {h}",
                    f"static unsigned char {frame_name}_bits[] = {{"
                ]
                hex_vals = [f"0x{b:02x}" for b in frame]
                for j in range(0, len(hex_vals), 12):
                    lines.append("   " + ", ".join(hex_vals[j:j+12]) + ",")
                lines.append("};")
                with open(frame_path, 'w') as f:
                    f.write("\n".join(lines))
        return os.path.getsize(filepath) if len(frames) == 1 else sum(
            os.path.getsize(f"{base}_{i:04d}{ext}") for i in range(len(frames)))

    def _write_c_header(self, frames, w, h, bit_depth, fps, filename, filepath):
        NAME = filename.upper()
        bytes_per_frame = len(frames[0]) if frames else 0
        total_bytes = bytes_per_frame * len(frames)

        lines = [
            "// SIDKIT Export - Generated by Koshi Nodes",
            f"// Generated: {datetime.now().isoformat()}",
            f"// Target: OLED {w}x{h} {bit_depth}-bit",
            f"// Frames: {len(frames)} @ {fps} FPS",
            f"// Memory: {total_bytes:,} bytes",
            "", "#pragma once", "#include <stdint.h>", "",
            f"#define {NAME}_WIDTH {w}",
            f"#define {NAME}_HEIGHT {h}",
            f"#define {NAME}_FRAME_COUNT {len(frames)}",
            f"#define {NAME}_FPS {fps}",
            f"#define {NAME}_BIT_DEPTH {bit_depth}",
            f"#define {NAME}_BYTES_PER_FRAME {bytes_per_frame}", "",
        ]

        for i, frame in enumerate(frames):
            lines.append(f"const uint8_t PROGMEM {filename}_frame_{i}[{len(frame)}] = {{")
            hex_vals = [f"0x{b:02X}" for b in frame]
            for j in range(0, len(hex_vals), 16):
                lines.append("    " + ", ".join(hex_vals[j:j+16]) + ",")
            lines.append("};")
            lines.append("")

        lines.append(f"const uint8_t* const {filename}_frames[{len(frames)}] = {{")
        for i in range(len(frames)):
            lines.append(f"    {filename}_frame_{i},")
        lines.append("};")

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
        return os.path.getsize(filepath)

    def _write_binary(self, frames, filepath):
        with open(filepath, 'wb') as f:
            for frame in frames:
                f.write(frame.tobytes())
        return os.path.getsize(filepath)

    def _to_hex_string(self, frames, limit=500):
        all_bytes = b''.join(f.tobytes() for f in frames)
        hex_str = ' '.join(f'{b:02X}' for b in all_bytes[:limit])
        if len(all_bytes) > limit:
            hex_str += f" ... ({len(all_bytes)} bytes total)"
        return hex_str

    def export(self, images, format, bit_depth, screen_preset, filename,
               custom_width=256, custom_height=128, fps=30, threshold=0.5,
               invert=False, output_path="", show_hex=False):

        width, height = self._get_dimensions(screen_preset, custom_width, custom_height)
        bit_depth_int = self._parse_bit_depth(bit_depth)

        if output_path and os.path.isdir(output_path):
            out_dir = output_path
        else:
            out_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "output", "sidkit")
        os.makedirs(out_dir, exist_ok=True)

        filename = "".join(c for c in filename if c.isalnum() or c in "_-") or "sidkit_export"

        images_np = images.cpu().numpy()
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]

        packed_frames = []
        for i in range(images_np.shape[0]):
            frame = images_np[i]
            if frame.max() > 1.0:
                frame = frame / 255.0
            gray = self._to_grayscale(frame)
            gray = self._resize_frame(gray, width, height)
            if invert:
                gray = 1.0 - gray
            quantized = self._quantize(gray, bit_depth_int, threshold)
            packed = self._pack_frame(quantized, bit_depth_int)
            packed_frames.append(packed)

        ext_map = {"sidv": ".sidv", "xbm": ".xbm", "c_header": ".h", "binary": ".bin"}
        filepath = os.path.join(out_dir, f"{filename}{ext_map.get(format, '.sidv')}")

        if format == "sidv":
            file_size = self._write_sidv(packed_frames, width, height, bit_depth_int, fps, filepath)
        elif format == "xbm":
            file_size = self._write_xbm(packed_frames, width, height, filename, filepath)
        elif format == "c_header":
            file_size = self._write_c_header(packed_frames, width, height, bit_depth_int,
                                              fps, filename, filepath)
        else:
            file_size = self._write_binary(packed_frames, filepath)

        hex_data = self._to_hex_string(packed_frames) if show_hex else ""

        return (filepath, hex_data, len(packed_frames), file_size)


NODE_CLASS_MAPPINGS = {"SIDKIT_Export": SIDKITExport}
NODE_DISPLAY_NAME_MAPPINGS = {"SIDKIT_Export": "░▒░ SIDKIT Export"}
