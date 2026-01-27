"""
Chromatic Aberration Effect for ComfyUI
Based on alien.js by Patrick Schroen
https://github.com/alienkitty/alien.js
MIT License
"""

import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False

try:
    from scipy.ndimage import map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KoshiChromaticAberration:
    """
    Chromatic aberration effect - separates RGB channels with offset.
    Based on alien.js ChromaticAberrationShader by Patrick Schroen.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self.use_gpu = MODERNGL_AVAILABLE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "red_offset": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "green_offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "blue_offset": ("FLOAT", {"default": -1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Koshi/Effects"
    OUTPUT_NODE = True

    def _init_gpu(self):
        """Initialize ModernGL context and shader."""
        if self.ctx is not None:
            return True

        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception as e:
            logger.debug(f"[Koshi] Failed to create GL context: {e}")
            self.use_gpu = False
            return False

        try:
            shader_path = Path(__file__).parent.parent.parent / "shaders" / "chromatic_aberration.glsl"
            with open(shader_path, 'r') as f:
                frag = f.read()

            vert = """
            #version 330
            in vec2 in_vert;
            out vec2 vUv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                vUv = (in_vert + 1.0) * 0.5;
            }
            """

            self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

            verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
            vbo = self.ctx.buffer(verts.tobytes())
            self.vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert')
            return True

        except Exception as e:
            logger.debug(f"[Koshi] Shader error: {e}")
            self.use_gpu = False
            return False

    def _apply_gpu(self, img_np, intensity, red_offset, green_offset, blue_offset):
        """GPU path using ModernGL."""
        h, w = img_np.shape[:2]

        # Ensure RGBA
        if img_np.shape[2] == 3:
            img_np = np.dstack([img_np, np.ones((h, w), dtype=np.float32)])

        tex = self.ctx.texture((w, h), 4, img_np.astype('f4').tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((w, h), 4, dtype='f4')])
        fbo.use()

        tex.use(location=0)
        self.prog['tMap'].value = 0
        self.prog['uIntensity'].value = intensity
        self.prog['uRedOffset'].value = red_offset
        self.prog['uGreenOffset'].value = green_offset
        self.prog['uBlueOffset'].value = blue_offset

        self.vao.render(moderngl.TRIANGLE_STRIP)

        data = fbo.read(components=4, dtype='f4')
        result = np.frombuffer(data, dtype=np.float32).reshape((h, w, 4))
        result = np.flip(result, axis=0)[:, :, :3].copy()

        tex.release()
        fbo.release()

        return result

    def _apply_cpu(self, img_np, intensity, red_offset, green_offset, blue_offset):
        """CPU fallback using coordinate remapping."""
        h, w = img_np.shape[:2]
        result = np.zeros_like(img_np)

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize to 0-1
        y_norm = y / h
        x_norm = x / w

        for c, offset in enumerate([red_offset, green_offset, blue_offset]):
            scale = 1.0 + 0.001 * offset * intensity
            shift = 0.001 * offset * intensity / 2.0

            # Transform UV coordinates
            new_x = x_norm * scale - shift
            new_y = y_norm * scale - shift

            # Convert back to pixel coordinates
            new_x = new_x * w
            new_y = new_y * h

            if SCIPY_AVAILABLE:
                result[:, :, c] = map_coordinates(
                    img_np[:, :, c],
                    [new_y, new_x],
                    order=1,
                    mode='reflect'
                )
            else:
                # Simple nearest neighbor fallback
                nx = np.clip(new_x.astype(int), 0, w - 1)
                ny = np.clip(new_y.astype(int), 0, h - 1)
                result[:, :, c] = img_np[ny, nx, c]

        return np.clip(result, 0, 1)

    def apply(self, image, intensity, red_offset, green_offset, blue_offset):
        if self.use_gpu:
            self._init_gpu()

        results = []
        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()

            if self.use_gpu and self.ctx is not None:
                result = self._apply_gpu(img_np, intensity, red_offset, green_offset, blue_offset)
            else:
                result = self._apply_cpu(img_np, intensity, red_offset, green_offset, blue_offset)

            results.append(result)

        return (torch.from_numpy(np.stack(results)).float().to(image.device),)


NODE_CLASS_MAPPINGS = {
    "Koshi_ChromaticAberration": KoshiChromaticAberration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_ChromaticAberration": "Koshi Chromatic Aberration",
}
