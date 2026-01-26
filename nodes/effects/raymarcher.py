"""
Dithering Raymarcher Node for ComfyUI
Executes a raymarching shader with controllable dithering grain

Dithering patterns based on glsl-dither by Hugh Kennedy (MIT)
https://github.com/hughsk/glsl-dither
"""

import torch
import numpy as np
import os
from pathlib import Path

# Optional ModernGL import (GPU required for raymarching)
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    print("[Koshi] ModernGL not available - DitheringRaymarcher will not function")

class DitheringRaymarcher:
    """
    A ComfyUI node that renders a raymarched scene with dithering effects.
    Features multiple dithering patterns (Blue Noise, Bayer 2x2/4x4/8x8, Random),
    grain control, and mono output option.
    """

    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None
        self.blue_noise_texture = None
        self.current_size = (0, 0)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64
                }),
                "time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01
                }),
                "shape": (["Torus", "Cube", "Sphere", "Dodecahedron", "Tetrahedron"], {
                    "default": "Torus"
                }),
                "pattern_type": (["Blue Noise", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Random"], {
                    "default": "Bayer 8x8"
                }),
                "grain_amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "mono_output": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "Koshi/Effects"

    def cleanup_resources(self):
        """Explicitly cleanup OpenGL resources"""
        resources = [
            ('fbo', self.fbo),
            ('vao', self.vao),
            ('prog', self.prog),
            ('blue_noise_texture', self.blue_noise_texture),
            ('ctx', self.ctx)
        ]

        for name, resource in resources:
            if resource is not None:
                try:
                    resource.release()
                except Exception as e:
                    print(f"Warning: Failed to release {name}: {e}")
                setattr(self, name, None)

        self.current_size = (0, 0)

    def initialize_gl(self, width, height):
        """Initialize or reinitialize OpenGL context and shader"""
        try:
            # Create standalone context
            if self.ctx is None:
                self.ctx = moderngl.create_standalone_context()

            # Load shader code
            shader_path = Path(__file__).parent.parent.parent / "shaders" / "dithering_raymarcher.glsl"
            if not shader_path.exists():
                raise FileNotFoundError(f"Shader file not found: {shader_path}")

            with open(shader_path, 'r') as f:
                fragment_shader = f.read()

            # Vertex shader for fullscreen quad
            vertex_shader = """
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
            """

            # Compile shader program
            if self.prog is not None:
                try:
                    self.prog.release()
                except:
                    pass

            self.prog = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except moderngl.Error as e:
            self.cleanup_resources()
            raise RuntimeError(f"Shader compilation failed: {e}")
        except Exception as e:
            self.cleanup_resources()
            raise RuntimeError(f"GL initialization failed: {e}")

        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')

        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert')

        # Create or recreate framebuffer if size changed
        if self.current_size != (width, height):
            if self.fbo is not None:
                self.fbo.release()

            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )
            self.current_size = (width, height)

        # Load blue noise texture
        if self.blue_noise_texture is None:
            self.blue_noise_texture = self.create_blue_noise_texture()

    def create_blue_noise_texture(self):
        """
        Create white noise texture as blue noise approximation.

        NOTE: This is NOT true blue noise - it's white noise with a fixed seed.
        True blue noise requires sophisticated generation algorithms (Poisson disk
        sampling, void-and-cluster, or pre-computed LUTs). For production quality,
        consider loading a pre-generated blue noise texture file.

        White noise is used here as it's computationally free and provides basic
        dithering, though it lacks blue noise's perceptual advantages (low-frequency
        suppression and more pleasing visual distribution).
        """
        size = 64  # Larger texture for better tiling
        np.random.seed(42)  # Consistent pattern
        noise = np.random.random((size, size)).astype('f4')

        texture = self.ctx.texture((size, size), 1, noise.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = True
        texture.repeat_y = True

        return texture

    def render(self, width, height, time, shape, pattern_type, grain_amount, mono_output):
        """Render the shader and return as ComfyUI image tensor"""
        # Check ModernGL availability (required for raymarching)
        if not MODERNGL_AVAILABLE:
            raise RuntimeError(
                "ModernGL is required for DitheringRaymarcher but is not installed. "
                "Install it with: pip install moderngl\n"
                "Note: ModernGL requires OpenGL support, which may not be available on M1/M2 Macs."
            )

        # Input validation
        width = max(64, min(4096, int(width)))
        height = max(64, min(4096, int(height)))
        time = max(0.0, float(time))
        grain_amount = max(0.0, min(1.0, float(grain_amount)))

        # Initialize or reinitialize GL context
        self.initialize_gl(width, height)

        # Map shape type string to integer
        shape_map = {
            "Torus": 0,
            "Cube": 1,
            "Sphere": 2,
            "Dodecahedron": 3,
            "Tetrahedron": 4
        }
        shape_int = shape_map.get(shape, 0)

        # Map pattern type string to integer
        pattern_map = {
            "Blue Noise": 0,
            "Bayer 2x2": 1,
            "Bayer 4x4": 2,
            "Bayer 8x8": 3,
            "Random": 4
        }
        pattern_int = pattern_map.get(pattern_type, 3)

        # Set uniforms
        self.prog['iResolution'].value = (width, height)
        self.prog['iTime'].value = time
        self.prog['shapeType'].value = shape_int
        self.prog['patternType'].value = pattern_int
        self.prog['grainAmount'].value = grain_amount
        self.prog['monoOutput'].value = mono_output

        # Bind blue noise texture (used for pattern 0)
        self.blue_noise_texture.use(location=0)
        self.prog['iChannel0'].value = 0

        # Render to framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read pixels
        data = self.fbo.read(components=4, dtype='f1')
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

        # Flip vertically (OpenGL coordinate system)
        image = np.flip(image, axis=0).copy()

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to ComfyUI format: (batch, height, width, channels)
        # Remove alpha channel and add batch dimension
        image = image[:, :, :3]
        image = torch.from_numpy(image).unsqueeze(0)

        return (image,)

    def __del__(self):
        """Cleanup OpenGL resources"""
        if self.fbo is not None:
            self.fbo.release()
        if self.vao is not None:
            self.vao.release()
        if self.prog is not None:
            self.prog.release()
        if self.blue_noise_texture is not None:
            self.blue_noise_texture.release()
        if self.ctx is not None:
            self.ctx.release()


# Register node
NODE_CLASS_MAPPINGS = {
    "KoshiDitheringRaymarcher": DitheringRaymarcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoshiDitheringRaymarcher": "Dithering Raymarcher 🌊"
}
