"""
SIDKIT Export Node - Export dithered frames to SIDKIT formats
Outputs: .sidv (Teensy video), .xbm (standard), .h (C header)

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
from pathlib import Path
from datetime import datetime


class SIDKITExport:
    """
    Export dithered images/video to SIDKIT-compatible formats.
    Supports .sidv (Teensy), .xbm (standard), and .h (C header).
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    def __init__(self):
        self.output_dir = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bit_depth": (["1-bit (mono)", "2-bit (4 levels)", "4-bit (16 levels)"], {
                    "default": "1-bit (mono)"
                }),
                "target_width": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 512,
                    "step": 8
                }),
                "target_height": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 256,
                    "step": 8
                }),
                "fps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 60,
                    "step": 1
                }),
                "output_format": (["sidv", "xbm", "c_header", "binary"], {
                    "default": "sidv"
                }),
                "filename": ("STRING", {
                    "default": "sidkit_output"
                }),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": ""
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("file_path", "frame_count", "file_size_bytes")
    FUNCTION = "export_sidkit"
    CATEGORY = "Koshi/Export"
    OUTPUT_NODE = True

    def _parse_bit_depth(self, bit_depth_str):
        """Parse bit depth from dropdown string."""
        if "1-bit" in bit_depth_str:
            return 1
        elif "2-bit" in bit_depth_str:
            return 2
        elif "4-bit" in bit_depth_str:
            return 4
        return 1

    def _resize_frame(self, frame, target_width, target_height):
        """Resize frame to target dimensions using nearest neighbor."""
        h, w = frame.shape[:2]
        if h == target_height and w == target_width:
            return frame

        y_ratio = h / target_height
        x_ratio = w / target_width

        y_indices = (np.arange(target_height) * y_ratio).astype(int)
        x_indices = (np.arange(target_width) * x_ratio).astype(int)

        y_indices = np.clip(y_indices, 0, h - 1)
        x_indices = np.clip(x_indices, 0, w - 1)

        return frame[y_indices][:, x_indices]

    def _to_grayscale(self, frame):
        """Convert RGB frame to grayscale."""
        if len(frame.shape) == 2:
            return frame
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]

    def _quantize(self, gray, bit_depth, threshold=0.5):
        """Quantize grayscale to specified bit depth."""
        if bit_depth == 1:
            return (gray > threshold).astype(np.uint8)
        elif bit_depth == 2:
            return np.clip((gray * 4).astype(np.uint8), 0, 3)
        elif bit_depth == 4:
            return np.clip((gray * 16).astype(np.uint8), 0, 15)
        return (gray > threshold).astype(np.uint8)

    def _pack_1bit(self, frame):
        """Pack 1-bit frame to bytes (8 pixels per byte, MSB first)."""
        h, w = frame.shape
        pad_w = (8 - w % 8) % 8
        if pad_w > 0:
            frame = np.pad(frame, ((0, 0), (0, pad_w)), constant_values=0)

        w_padded = frame.shape[1]
        packed = np.zeros((h, w_padded // 8), dtype=np.uint8)

        for bit in range(8):
            packed |= (frame[:, bit::8] << (7 - bit))

        return packed.flatten()

    def _pack_2bit(self, frame):
        """Pack 2-bit frame to bytes (4 pixels per byte)."""
        h, w = frame.shape
        pad_w = (4 - w % 4) % 4
        if pad_w > 0:
            frame = np.pad(frame, ((0, 0), (0, pad_w)), constant_values=0)

        w_padded = frame.shape[1]
        packed = np.zeros((h, w_padded // 4), dtype=np.uint8)

        for i in range(4):
            packed |= (frame[:, i::4] & 0x03) << (6 - i * 2)

        return packed.flatten()

    def _pack_4bit(self, frame):
        """Pack 4-bit frame to bytes (2 pixels per byte, SSD1363 nibble order)."""
        h, w = frame.shape
        if w % 2:
            frame = np.pad(frame, ((0, 0), (0, 1)), constant_values=0)

        packed = (frame[:, 0::2] << 4) | (frame[:, 1::2] & 0x0F)
        return packed.flatten()

    def _pack_frame(self, frame, bit_depth):
        """Pack frame according to bit depth."""
        if bit_depth == 1:
            return self._pack_1bit(frame)
        elif bit_depth == 2:
            return self._pack_2bit(frame)
        elif bit_depth == 4:
            return self._pack_4bit(frame)
        return self._pack_1bit(frame)

    def _write_sidv(self, frames, width, height, bit_depth, fps, filepath):
        """Write frames to .sidv format."""
        frame_count = len(frames)

        header = bytearray(16)
        header[0:4] = b'SIDK'
        header[4] = 1
        struct.pack_into('<H', header, 5, width)
        struct.pack_into('<H', header, 7, height)
        header[9] = bit_depth
        header[10] = fps
        struct.pack_into('<I', header, 11, frame_count)
        header[15] = 0

        with open(filepath, 'wb') as f:
            f.write(header)
            for frame in frames:
                f.write(frame.tobytes())

        return os.path.getsize(filepath)

    def _write_xbm(self, frames, width, height, filename, filepath):
        """Write frames to XBM format."""
        total_size = 0

        if len(frames) == 1:
            frame = frames[0]
            lines = [
                f"#define {filename}_width {width}",
                f"#define {filename}_height {height}",
                f"static unsigned char {filename}_bits[] = {{"
            ]

            hex_values = [f"0x{b:02x}" for b in frame]
            for i in range(0, len(hex_values), 12):
                lines.append("   " + ", ".join(hex_values[i:i+12]) + ",")

            lines.append("};")

            with open(filepath, 'w') as f:
                f.write("\n".join(lines))
            total_size = os.path.getsize(filepath)
        else:
            base, ext = os.path.splitext(filepath)
            for i, frame in enumerate(frames):
                frame_path = f"{base}_{i:04d}{ext}"
                frame_name = f"{filename}_{i:04d}"

                lines = [
                    f"#define {frame_name}_width {width}",
                    f"#define {frame_name}_height {height}",
                    f"static unsigned char {frame_name}_bits[] = {{"
                ]

                hex_values = [f"0x{b:02x}" for b in frame]
                for j in range(0, len(hex_values), 12):
                    lines.append("   " + ", ".join(hex_values[j:j+12]) + ",")

                lines.append("};")

                with open(frame_path, 'w') as f:
                    f.write("\n".join(lines))
                total_size += os.path.getsize(frame_path)

        return total_size

    def _write_c_header(self, frames, width, height, bit_depth, fps, filename, filepath):
        """Write frames to C header format for Teensy."""
        frame_count = len(frames)
        bytes_per_frame = len(frames[0]) if frames else 0
        total_bytes = bytes_per_frame * frame_count

        lines = [
            "// SIDKIT Export - Generated by Koshi Nodes",
            f"// Generated: {datetime.now().isoformat()}",
            f"// Target: SSD1363 {width}x{height} {bit_depth}-bit",
            f"// Frames: {frame_count} @ {fps} FPS = {frame_count/fps:.1f}s",
            f"// Memory: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)",
            "",
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"#define {filename.upper()}_WIDTH {width}",
            f"#define {filename.upper()}_HEIGHT {height}",
            f"#define {filename.upper()}_FRAME_COUNT {frame_count}",
            f"#define {filename.upper()}_FPS {fps}",
            f"#define {filename.upper()}_FRAME_DELAY_MS {1000 // fps}",
            f"#define {filename.upper()}_BIT_DEPTH {bit_depth}",
            f"#define {filename.upper()}_BYTES_PER_FRAME {bytes_per_frame}",
            "",
        ]

        for i, frame in enumerate(frames):
            lines.append(f"const uint8_t PROGMEM {filename}_frame_{i}[{len(frame)}] = {{")
            hex_values = [f"0x{b:02X}" for b in frame]
            for j in range(0, len(hex_values), 16):
                lines.append("    " + ", ".join(hex_values[j:j+16]) + ",")
            lines.append("};")
            lines.append("")

        lines.append(f"const uint8_t* const {filename}_frames[{frame_count}] = {{")
        for i in range(frame_count):
            lines.append(f"    {filename}_frame_{i},")
        lines.append("};")

        lines.append("")
        lines.append(self._generate_player_class(filename, width, height, bit_depth, fps))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))

        return os.path.getsize(filepath)

    def _generate_player_class(self, name, width, height, bit_depth, fps):
        """Generate C++ player class for Teensy."""
        NAME = name.upper()

        if bit_depth == 1:
            draw_code = f'''        const uint8_t* data = {name}_frames[frame];
        for (int y = 0; y < {NAME}_HEIGHT; y++) {{
            for (int x = 0; x < {NAME}_WIDTH; x += 8) {{
                uint8_t b = pgm_read_byte(data++);
                for (int bit = 0; bit < 8; bit++) {{
                    gfx.drawPixel(ox + x + bit, oy + y, (b >> (7 - bit)) & 1 ? 0xF : 0x0);
                }}
            }}
        }}'''
        elif bit_depth == 2:
            draw_code = f'''        const uint8_t* data = {name}_frames[frame];
        for (int y = 0; y < {NAME}_HEIGHT; y++) {{
            for (int x = 0; x < {NAME}_WIDTH; x += 4) {{
                uint8_t b = pgm_read_byte(data++);
                for (int p = 0; p < 4; p++) {{
                    uint8_t val = (b >> (6 - p * 2)) & 0x03;
                    gfx.drawPixel(ox + x + p, oy + y, val * 5);
                }}
            }}
        }}'''
        else:
            draw_code = f'''        const uint8_t* data = {name}_frames[frame];
        for (int y = 0; y < {NAME}_HEIGHT; y++) {{
            for (int x = 0; x < {NAME}_WIDTH; x += 2) {{
                uint8_t b = pgm_read_byte(data++);
                gfx.drawPixel(ox + x, oy + y, (b >> 4) & 0x0F);
                gfx.drawPixel(ox + x + 1, oy + y, b & 0x0F);
            }}
        }}'''

        return f'''// Animation player helper
class {name.title().replace("_", "")}Player {{
public:
    uint16_t frame = 0;
    uint32_t lastTime = 0;

    bool update() {{
        uint32_t now = millis();
        if (now - lastTime >= {NAME}_FRAME_DELAY_MS) {{
            lastTime = now;
            frame = (frame + 1) % {NAME}_FRAME_COUNT;
            return true;
        }}
        return false;
    }}

    template<typename GFX>
    void draw(GFX& gfx, int ox = 0, int oy = 0) {{
{draw_code}
    }}

    void reset() {{ frame = 0; lastTime = millis(); }}
    uint16_t getFrame() {{ return frame; }}
    void setFrame(uint16_t f) {{ frame = f % {NAME}_FRAME_COUNT; }}
}};'''

    def _write_binary(self, frames, filepath):
        """Write raw binary frames (no header)."""
        with open(filepath, 'wb') as f:
            for frame in frames:
                f.write(frame.tobytes())
        return os.path.getsize(filepath)

    def export_sidkit(self, images, bit_depth, target_width, target_height, fps,
                      output_format, filename, output_path="", threshold=0.5):
        """Export images to SIDKIT format."""

        bit_depth_int = self._parse_bit_depth(bit_depth)

        if output_path and os.path.isdir(output_path):
            out_dir = output_path
        else:
            out_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "output", "sidkit")

        os.makedirs(out_dir, exist_ok=True)

        filename = "".join(c for c in filename if c.isalnum() or c in "_-")
        if not filename:
            filename = "sidkit_output"

        images_np = images.cpu().numpy()
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]

        packed_frames = []
        for i in range(images_np.shape[0]):
            frame = images_np[i]

            if frame.max() > 1.0:
                frame = frame / 255.0

            gray = self._to_grayscale(frame)
            gray = self._resize_frame(gray, target_width, target_height)
            quantized = self._quantize(gray, bit_depth_int, threshold)
            packed = self._pack_frame(quantized, bit_depth_int)
            packed_frames.append(packed)

        ext_map = {
            "sidv": ".sidv",
            "xbm": ".xbm",
            "c_header": ".h",
            "binary": ".bin"
        }
        ext = ext_map.get(output_format, ".sidv")
        filepath = os.path.join(out_dir, f"{filename}{ext}")

        if output_format == "sidv":
            file_size = self._write_sidv(packed_frames, target_width, target_height,
                                         bit_depth_int, fps, filepath)
        elif output_format == "xbm":
            file_size = self._write_xbm(packed_frames, target_width, target_height,
                                        filename, filepath)
        elif output_format == "c_header":
            file_size = self._write_c_header(packed_frames, target_width, target_height,
                                             bit_depth_int, fps, filename, filepath)
        elif output_format == "binary":
            file_size = self._write_binary(packed_frames, filepath)
        else:
            file_size = self._write_sidv(packed_frames, target_width, target_height,
                                         bit_depth_int, fps, filepath)

        frame_count = len(packed_frames)

        return (filepath, frame_count, file_size)


NODE_CLASS_MAPPINGS = {
    "KoshiSIDKITExport": SIDKITExport
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoshiSIDKITExport": "SIDKIT Export"
}
