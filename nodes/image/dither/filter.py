"""
Image Dithering Filter - Hybrid GPU/CPU implementation
Patterns based on glsl-dither by Hugh Kennedy (MIT)
https://github.com/hughsk/glsl-dither

Auto-detects GPU (ModernGL) or falls back to CPU (NumPy).
"""

import torch
import numpy as np
from pathlib import Path

# Optional ModernGL import (GPU path)
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    print("[Koshi] ModernGL not available, using CPU fallback for dithering")

class ImageDitheringFilter:
    """
    A ComfyUI node that applies dithering effects to input images.
    Features color palette control, posterization, and pixel grid sizing.
    """
    COLOR = "#FF9F43"
    BGCOLOR = "#1a1a1a"

    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None
        self.input_texture = None
        self.current_size = (0, 0)
        self.use_gpu = MODERNGL_AVAILABLE

        # Pre-generate Bayer matrices for CPU path
        self.bayer_2x2 = self._generate_bayer_matrix(2)
        self.bayer_4x4 = self._generate_bayer_matrix(4)
        self.bayer_8x8 = self._generate_bayer_matrix(8)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI IMAGE input
                "pattern_type": (["Blue Noise", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Random"], {
                    "default": "Bayer 8x8"
                }),
                "use_original_colors": ("BOOLEAN", {
                    "default": False
                }),
                "color_steps": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 7,
                    "step": 1
                }),
                "pixel_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5
                }),
                "dither_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "color_back": ("STRING", {
                    "default": "#000000"
                }),
                "color_front": ("STRING", {
                    "default": "#ffffff"
                }),
                "color_highlight": ("STRING", {
                    "default": "#ffffff"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_dithering"
    CATEGORY = "Koshi/Image/Dither"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (0.0-1.0 range)"""
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                raise ValueError(f"Invalid hex color length: {len(hex_color)}")

            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        except (ValueError, IndexError) as e:
            print(f"Warning: Invalid hex color '{hex_color}', using black. Error: {e}")
            return (0.0, 0.0, 0.0)

    # ========================================
    # CPU-Based Dithering Methods
    # ========================================

    def _generate_bayer_matrix(self, n):
        """
        Recursively generate Bayer matrix of size n×n.
        Uses the recursive definition of ordered dithering matrices.
        """
        if n == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        else:
            smaller = self._generate_bayer_matrix(n // 2)
            return np.block([
                [4 * smaller,     4 * smaller + 2],
                [4 * smaller + 3, 4 * smaller + 1]
            ]) / (n * n)

    def _apply_bayer_dithering_cpu(self, image, matrix, threshold_multiplier=1.0):
        """
        Apply Bayer matrix dithering to image (CPU implementation).

        Args:
            image: NumPy array of shape (H, W, C) in range [0, 1]
            matrix: Bayer matrix (2x2, 4x4, or 8x8)
            threshold_multiplier: Scale factor for dithering intensity

        Returns:
            Dithered binary image (0 or 1 values)
        """
        height, width, channels = image.shape
        matrix_size = matrix.shape[0]

        # Convert to grayscale for thresholding
        if channels == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image[:, :, 0]

        # Tile the Bayer matrix to match image dimensions
        tile_y = (height + matrix_size - 1) // matrix_size
        tile_x = (width + matrix_size - 1) // matrix_size
        threshold_map = np.tile(matrix, (tile_y, tile_x))[:height, :width]

        # Apply dithering with intensity control
        threshold_map = threshold_map * threshold_multiplier
        dithered = (gray > threshold_map).astype(np.float32)

        return dithered

    def _posterize(self, image, levels):
        """Posterize image to specified number of levels."""
        if levels <= 1:
            return image
        steps = levels - 1
        return np.floor(image * steps + 0.5) / steps

    def apply_dithering_cpu(self, image, pattern_type, use_original_colors, color_steps,
                           pixel_size, dither_intensity, color_back, color_front, color_highlight):
        """
        CPU-based dithering implementation (fallback when ModernGL unavailable).
        Optimized NumPy operations for reasonable performance.
        """
        # Convert to numpy and ensure float32 [0, 1] range
        img_np = image.cpu().numpy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        height, width, channels = img_np.shape

        # Apply pixel grid effect if needed
        if pixel_size > 1.0:
            scale = int(pixel_size)
            new_h, new_w = height // scale, width // scale
            # Downsample
            img_np = img_np[::scale, ::scale, :]
            # Nearest neighbor upscale
            img_np = np.repeat(np.repeat(img_np, scale, axis=0), scale, axis=1)
            img_np = img_np[:height, :width, :]  # Crop to original size

        # Select dithering matrix
        if pattern_type == "Bayer 2x2":
            matrix = self.bayer_2x2
        elif pattern_type == "Bayer 4x4":
            matrix = self.bayer_4x4
        elif pattern_type == "Bayer 8x8":
            matrix = self.bayer_8x8
        elif pattern_type == "Random":
            matrix_size = 8
            matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        else:  # Blue Noise (approximated as random for CPU)
            matrix_size = 8
            np.random.seed(42)
            matrix = np.random.rand(matrix_size, matrix_size).astype(np.float32)

        # Apply dithering
        dithered_mask = self._apply_bayer_dithering_cpu(img_np, matrix, dither_intensity)

        if use_original_colors:
            # Posterize original colors
            output = self._posterize(img_np, color_steps)
            # Apply dithering as modulation
            output = output * (0.8 + 0.4 * dithered_mask[:, :, np.newaxis])
            output = np.clip(output, 0, 1)
        else:
            # Use custom color palette
            rgb_back = np.array(self.hex_to_rgb(color_back), dtype=np.float32)
            rgb_front = np.array(self.hex_to_rgb(color_front), dtype=np.float32)
            rgb_highlight = np.array(self.hex_to_rgb(color_highlight), dtype=np.float32)

            # Create 3-level output based on dithered mask
            gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

            output = np.zeros_like(img_np)
            mask_dark = (gray < 0.33) | ((gray < 0.66) & (dithered_mask < 0.5))
            mask_highlight = gray > 0.66

            output[mask_dark] = rgb_back
            output[~mask_dark & ~mask_highlight] = rgb_front
            output[mask_highlight] = rgb_highlight

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output).unsqueeze(0)
        return (output_tensor,)

    def cleanup_resources(self):
        """Explicitly cleanup OpenGL resources"""
        resources = [
            ('input_texture', self.input_texture),
            ('fbo', self.fbo),
            ('vao', self.vao),
            ('prog', self.prog),
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
            if self.ctx is None:
                self.ctx = moderngl.create_standalone_context()

            # Load shader code
            shader_path = Path(__file__).parent.parent.parent.parent / "shaders" / "image_dithering_filter.glsl"
            if not shader_path.exists():
                raise FileNotFoundError(f"Shader file not found: {shader_path}")

            with open(shader_path, 'r') as f:
                fragment_shader = f.read()

            # Vertex shader for fullscreen quad
            vertex_shader = """
            #version 330
            in vec2 in_vert;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_uv = (in_vert + 1.0) * 0.5;
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

    def apply_dithering(self, image, pattern_type, use_original_colors, color_steps,
                       pixel_size, dither_intensity, color_back="#000000",
                       color_front="#ffffff", color_highlight="#ffffff"):
        """Apply dithering filter to input image and return as ComfyUI image tensor"""

        # Input validation
        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError("image must be a valid torch.Tensor")

        if len(image.shape) not in [3, 4]:
            raise ValueError(f"image must be 3D or 4D tensor, got shape {image.shape}")

        # ComfyUI images are in format: (batch, height, width, channels)
        # Take first image from batch
        if len(image.shape) == 4:
            if image.shape[0] == 0:
                raise ValueError("image batch cannot be empty")
            image = image[0]

        height, width, channels = image.shape

        # Validate and clamp parameters
        color_steps = max(1, min(7, int(color_steps)))
        pixel_size = max(0.5, min(20.0, float(pixel_size)))
        dither_intensity = max(0.0, min(1.0, float(dither_intensity)))

        # Route to CPU or GPU path based on availability
        if not self.use_gpu or not MODERNGL_AVAILABLE:
            print("[Koshi] Using CPU dithering fallback (ModernGL not available or disabled)")
            return self.apply_dithering_cpu(
                image, pattern_type, use_original_colors, color_steps,
                pixel_size, dither_intensity, color_back, color_front, color_highlight
            )

        # GPU path: Initialize GL context
        self.initialize_gl(width, height)

        # Convert image to numpy and ensure it's in the right format
        img_np = image.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Create or update input texture
        if self.input_texture is not None:
            self.input_texture.release()

        self.input_texture = self.ctx.texture((width, height), 3, img_np.tobytes())
        self.input_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Map pattern type string to integer
        pattern_map = {
            "Blue Noise": 0,
            "Bayer 2x2": 1,
            "Bayer 4x4": 2,
            "Bayer 8x8": 3,
            "Random": 4
        }
        pattern_int = pattern_map.get(pattern_type, 3)

        # Parse color values
        rgb_back = self.hex_to_rgb(color_back)
        rgb_front = self.hex_to_rgb(color_front)
        rgb_highlight = self.hex_to_rgb(color_highlight)

        # Set uniforms
        self.prog['iResolution'].value = (width, height)
        self.prog['patternType'].value = pattern_int
        self.prog['colorBack'].value = rgb_back
        self.prog['colorFront'].value = rgb_front
        self.prog['colorHighlight'].value = rgb_highlight
        self.prog['useOriginalColors'].value = use_original_colors
        self.prog['colorSteps'].value = color_steps
        self.prog['pixelSize'].value = pixel_size
        self.prog['ditherIntensity'].value = dither_intensity

        # Bind input texture
        self.input_texture.use(location=0)
        self.prog['inputImage'].value = 0

        # Render to framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read pixels
        data = self.fbo.read(components=4, dtype='f1')
        output_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

        # Flip vertically (OpenGL coordinate system)
        output_image = np.flip(output_image, axis=0).copy()

        # Convert to float32 and normalize to [0, 1]
        output_image = output_image.astype(np.float32) / 255.0

        # Remove alpha channel and add batch dimension
        output_image = output_image[:, :, :3]
        output_image = torch.from_numpy(output_image).unsqueeze(0)

        return (output_image,)

    def __del__(self):
        """Cleanup OpenGL resources"""
        if self.input_texture is not None:
            self.input_texture.release()
        if self.fbo is not None:
            self.fbo.release()
        if self.vao is not None:
            self.vao.release()
        if self.prog is not None:
            self.prog.release()
        if self.ctx is not None:
            self.ctx.release()


# Register node
NODE_CLASS_MAPPINGS = {
    "Koshi_DitheringFilter": ImageDitheringFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_DitheringFilter": "░▒░ KN Dithering Filter"
}
