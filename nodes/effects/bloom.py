"""
Bloom Shader Node for ComfyUI
Implements a high-quality "Unreal-style" bloom effect.
GPU path uses ModernGL, CPU fallback uses scipy/numpy.
"""

import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Optional ModernGL import
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    logger.debug("[Koshi] ModernGL not available - Bloom will use CPU fallback")

# Optional scipy for CPU fallback
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class BloomShaderNode:
    """
    A ComfyUI node that applies a high-quality Bloom effect.
    Uses a multi-pass approach (Downsample/Upsample chain) for natural looking glow.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo_cache = {} # Cache FBOs by size
        self.current_size = (0, 0)
        self.use_gpu = MODERNGL_AVAILABLE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "radius": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_bloom"
    CATEGORY = "Koshi/Effects"
    OUTPUT_NODE = True

    def cleanup_resources(self):
        """Explicitly cleanup OpenGL resources"""
        if self.ctx:
            for fbo in self.fbo_cache.values():
                fbo.release()
            self.fbo_cache.clear()
            
            if self.vao: self.vao.release()
            if self.prog: self.prog.release()
            
            # We don't release context as it might be shared or expensive to recreate constantly
            # but for standalone context it's usually fine to keep it until node deletion
            # self.ctx.release() 

    def initialize_gl(self):
        """Initialize OpenGL context and shader"""
        if self.ctx is None:
            try:
                self.ctx = moderngl.create_standalone_context()
            except Exception as e:
                print(f"[Koshi] Failed to create GL context: {e}")
                self.use_gpu = False
                return

        if self.prog is None:
            try:
                shader_path = Path(__file__).parent.parent.parent / "shaders" / "bloom.glsl"
                if not shader_path.exists():
                    raise FileNotFoundError(f"Shader file not found: {shader_path}")

                with open(shader_path, 'r') as f:
                    fragment_shader = f.read()

                vertex_shader = """
                #version 330
                in vec2 in_vert;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_uv = (in_vert + 1.0) * 0.5;
                }
                """

                self.prog = self.ctx.program(
                    vertex_shader=vertex_shader,
                    fragment_shader=fragment_shader
                )

                # Quad geometry
                vertices = np.array([
                    -1.0, -1.0,
                     1.0, -1.0,
                    -1.0,  1.0,
                     1.0,  1.0,
                ], dtype='f4')
                vbo = self.ctx.buffer(vertices.tobytes())
                self.vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert')

            except Exception as e:
                print(f"[Koshi] Shader compilation failed: {e}")
                self.use_gpu = False

    def get_fbo(self, width, height):
        """Get or create FBO for specific size"""
        key = (width, height)
        if key not in self.fbo_cache:
            texture = self.ctx.texture((width, height), 4, dtype='f2') # Float16 is usually enough for HDR
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            fbo = self.ctx.framebuffer(color_attachments=[texture])
            self.fbo_cache[key] = fbo
        return self.fbo_cache[key]

    def _apply_bloom_cpu(self, image, threshold, intensity, radius):
        """CPU fallback using gaussian blur for bloom effect."""
        batch_results = []

        for img_tensor in image:
            img_np = img_tensor.cpu().numpy()

            # Convert to grayscale for brightness detection
            gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

            # Extract bright areas (threshold)
            bright_mask = np.clip((gray - threshold) / (1.0 - threshold + 0.001), 0, 1)
            bright = img_np * bright_mask[:, :, np.newaxis]

            # Apply blur (simulate bloom spread)
            blur_radius = max(1, int(radius * min(img_np.shape[:2]) * 0.1))

            if SCIPY_AVAILABLE:
                blurred = np.zeros_like(bright)
                for c in range(3):
                    blurred[:, :, c] = gaussian_filter(bright[:, :, c], sigma=blur_radius)
            else:
                # Simple box blur fallback
                from numpy.lib.stride_tricks import sliding_window_view
                k = blur_radius * 2 + 1
                padded = np.pad(bright, ((blur_radius, blur_radius), (blur_radius, blur_radius), (0, 0)), mode='reflect')
                blurred = np.zeros_like(bright)
                for c in range(3):
                    windows = sliding_window_view(padded[:, :, c], (k, k))
                    blurred[:, :, c] = windows[::1, ::1].mean(axis=(2, 3))[:bright.shape[0], :bright.shape[1]]

            # Combine original with bloom
            result = img_np + blurred * intensity
            result = np.clip(result, 0, 1)
            batch_results.append(result)

        return (torch.from_numpy(np.stack(batch_results)).float(),)

    def apply_bloom(self, image, threshold, intensity, radius):
        if not self.use_gpu:
            logger.debug("[Koshi] GPU not available, using CPU bloom")
            return self._apply_bloom_cpu(image, threshold, intensity, radius)

        self.initialize_gl()
        if not self.use_gpu:
            return (image,)

        # Process batch
        batch_results = []
        
        # Ensure image is float32
        if image.dtype != torch.float32:
            image = image.float()

        for img_tensor in image:
            # Convert to numpy (H, W, C)
            img_np = img_tensor.cpu().numpy()
            height, width, channels = img_np.shape
            
            # Pad alpha if needed
            if channels == 3:
                img_np = np.dstack([img_np, np.ones((height, width), dtype=np.float32)])
            
            # Upload input texture
            input_tex = self.ctx.texture((width, height), 4, img_np.astype('f4').tobytes())
            input_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # --- Pass 1: Prefilter (Threshold) ---
            # Render to half resolution for efficiency
            w, h = width // 2, height // 2
            fbo_mips = []
            
            # Prefilter pass
            fbo_pre = self.get_fbo(w, h)
            fbo_pre.use()
            input_tex.use(location=0)
            self.prog['image'].value = 0
            self.prog['mode'].value = 0 # Prefilter
            self.prog['threshold'].value = threshold
            self.prog['knee'].value = 0.5 # Hardcoded knee for now
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            fbo_mips.append(fbo_pre)
            
            # --- Pass 2: Downsample Chain ---
            # Create 5 mip levels
            mip_count = 5
            current_tex = fbo_pre.color_attachments[0]
            
            for i in range(mip_count):
                w, h = max(1, w // 2), max(1, h // 2)
                fbo_down = self.get_fbo(w, h)
                fbo_down.use()
                
                current_tex.use(location=0)
                self.prog['image'].value = 0
                self.prog['mode'].value = 1 # Downsample
                self.vao.render(moderngl.TRIANGLE_STRIP)
                
                fbo_mips.append(fbo_down)
                current_tex = fbo_down.color_attachments[0]
                
            # --- Pass 3: Upsample Chain ---
            # Blend up the chain
            # Start from the last mip
            current_tex = fbo_mips[-1].color_attachments[0]
            
            # We need to blend additively. ModernGL supports blending.
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
            
            # Iterate backwards from second to last to first
            for i in range(len(fbo_mips) - 2, -1, -1):
                fbo_target = fbo_mips[i]
                fbo_target.use()
                
                current_tex.use(location=0)
                self.prog['image'].value = 0
                self.prog['mode'].value = 2 # Upsample
                self.prog['intensity'].value = 1.0 # Accumulate
                
                self.vao.render(moderngl.TRIANGLE_STRIP)
                
                current_tex = fbo_target.color_attachments[0]
                
            self.ctx.disable(moderngl.BLEND)
            
            # --- Final Combine ---
            # We have the bloom result in fbo_mips[0] (which is half res)
            # We need to upscale it to full res and add to original
            
            # Create output FBO
            fbo_out = self.get_fbo(width, height)
            fbo_out.use()
            self.ctx.clear(0, 0, 0, 0)
            
            # Draw original image
            # We can just use a simple pass or blit, but let's use a simple shader pass if we had one
            # Or just draw a quad with the input texture
            # Actually, let's just do the combine in Python (numpy) to save a shader pass complexity
            # OR, render the bloom texture to a numpy array and add it to original on CPU
            
            # Let's read the bloom texture
            bloom_tex = fbo_mips[0].color_attachments[0]
            
            # We need to upsample the final bloom texture to full res
            # We can do one last upsample render to fbo_out
            fbo_out.use()
            bloom_tex.use(location=0)
            self.prog['image'].value = 0
            self.prog['mode'].value = 2 # Upsample
            self.vao.render(moderngl.TRIANGLE_STRIP)
            
            # Read bloom result
            data = fbo_out.read(components=4, dtype='f2') # Read float16
            bloom_np = np.frombuffer(data, dtype=np.float16).reshape((height, width, 4)).astype(np.float32)
            
            # Flip Y
            bloom_np = np.flip(bloom_np, axis=0)
            
            # Add to original (CPU side composition for simplicity and safety)
            # intensity controls how much bloom is added
            final_img = img_np[:, :, :3] + bloom_np[:, :, :3] * intensity
            final_img = np.clip(final_img, 0.0, 1.0)
            
            batch_results.append(final_img)
            
            # Cleanup textures for this run? 
            # We keep FBOs in cache, but textures attached to them are reused.
            # input_tex needs release
            input_tex.release()

        # Stack results
        output_tensor = torch.from_numpy(np.stack(batch_results)).float()
        return (output_tensor,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "KoshiBloomShader": BloomShaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoshiBloomShader": "Bloom Shader (Post-Process) ✨"
}
