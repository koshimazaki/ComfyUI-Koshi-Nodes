"""
Glitch Candies Generator - Procedural 3D shapes, fractals, and noise patterns.
Based on SIDKIT shader system with raymarched shapes and seamless loops.
"""

import torch
import numpy as np
import math


class KoshiGlitchCandies:
    """
    Generate procedural patterns and raymarched 3D shapes.
    Perfect for masks, backgrounds, and animated textures.
    """

    CATEGORY = "Koshi/Generators"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True

    PATTERNS = [
        # 2D Patterns
        "waves", "circles", "plasma", "voronoi", "checkerboard",
        "mandelbrot", "julia", "sierpinski", "swirl", "ripple",
        # Glitch Candies (seamless loops)
        "glitch_candies", "fbm_noise", "cell_noise", "distorted_grid",
        "height_map", "glitch_cubes",
        # Raymarched 3D
        "rm_cube", "rm_sphere", "rm_torus", "rm_octahedron",
        "rm_gyroid", "rm_menger",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "pattern": (cls.PATTERNS, {"default": "glitch_candies"}),
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "loop_frames": ("INT", {"default": 0, "min": 0, "max": 1000}),
                # Camera controls for raymarched 3D shapes
                "camera_distance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "rotation_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "rotation_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            }
        }

    def _hash(self, p):
        """2D hash function."""
        return np.fmod(np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7) * 43758.5453, 1.0)

    def _value_noise(self, p, t):
        """Value noise with time animation."""
        i = np.floor(p).astype(int)
        f = p - np.floor(p)
        f = f * f * (3.0 - 2.0 * f)
        
        def h(offset):
            ip = i + offset
            return self._hash(ip) + 0.2 * np.sin(t + self._hash(ip) * 6.28318)
        
        a = h(np.array([0, 0]))
        b = h(np.array([1, 0]))
        c = h(np.array([0, 1]))
        d = h(np.array([1, 1]))
        
        return (a * (1 - f[..., 0]) + b * f[..., 0]) * (1 - f[..., 1]) + \
               (c * (1 - f[..., 0]) + d * f[..., 0]) * f[..., 1]

    def _fbm(self, p, t, octaves=5):
        """Fractal Brownian Motion."""
        result = np.zeros_like(p[..., 0])
        amp = 0.5
        for i in range(octaves):
            result += amp * self._value_noise(p, t + i * 0.5)
            p = p * 2.0
            amp *= 0.5
        return result

    def _generate_pattern(self, width, height, pattern, time, scale, seed,
                          camera_dist=3.0, rot_x=0.0, rot_y=0.0):
        """Generate a single frame of the pattern."""
        np.random.seed(seed)

        # Create UV coordinates
        y, x = np.mgrid[0:height, 0:width]
        uv = np.stack([x / width, y / height], axis=-1)
        p = (uv - 0.5) * 2.0 * scale
        aspect = width / height
        
        if pattern == "waves":
            result = np.sin(uv[..., 0] * 20 + time) * np.sin(uv[..., 1] * 20 + time * 0.7)
            result += np.sin(np.linalg.norm(uv - 0.5, axis=-1) * 30 - time * 2) * 0.5
            result = result * 0.5 + 0.5
            
        elif pattern == "circles":
            d = np.linalg.norm(uv - 0.5, axis=-1)
            result = np.sin(d * 40 - time * 3) * 0.5 + 0.5
            
        elif pattern == "plasma":
            v1 = np.sin(uv[..., 0] * 10 + time)
            v2 = np.sin(uv[..., 1] * 10 + time * 1.2)
            v3 = np.sin((uv[..., 0] + uv[..., 1]) * 10 + time * 0.8)
            v4 = np.sin(np.linalg.norm(uv - 0.5, axis=-1) * 20 + time * 1.5)
            result = (v1 + v2 + v3 + v4) * 0.25 + 0.5
            
        elif pattern == "voronoi":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            vp = uv * 5.0
            ip = np.floor(vp).astype(int)
            fp = vp - np.floor(vp)
            
            min_d = np.ones((height, width)) * 8.0
            for j in range(-1, 2):
                for i in range(-1, 2):
                    g = np.array([i, j])
                    cell_id = ip + g
                    h = self._hash(cell_id)
                    o = 0.5 + 0.4 * np.sin(loop_time + h * 6.28318)
                    r = g[np.newaxis, np.newaxis, :] + o[..., np.newaxis] - fp
                    d = np.linalg.norm(r, axis=-1)
                    min_d = np.minimum(min_d, d)
            result = min_d
            
        elif pattern == "checkerboard":
            cp = np.floor(uv * 16 + time * 2)
            result = np.mod(cp[..., 0] + cp[..., 1], 2.0)
            
        elif pattern == "swirl":
            sp = uv - 0.5
            l = np.linalg.norm(sp, axis=-1)
            angle = 6.0 * np.arctan2(sp[..., 1], sp[..., 0]) + time * 2.0
            twist = 1.2
            offset = 1.0 / np.maximum(l, 0.001) ** twist + angle / 6.28318
            mid = np.clip(l ** twist, 0, 1)
            result = np.fmod(offset, 1.0) * mid
            
        elif pattern == "ripple":
            rp = uv - 0.5
            dist = np.linalg.norm(rp, axis=-1)
            result = np.sin(dist ** 1.7 * 20 - time * 3) * 0.5 + 0.5
            
        elif pattern == "glitch_candies":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            gp = (uv - 0.5) * 8.0 * scale
            
            # FBM
            fbm_val = self._fbm(gp, loop_time)
            
            # Voronoi
            ip = np.floor(gp).astype(int)
            fp = gp - np.floor(gp)
            min_d = np.ones((height, width))
            for j in range(-1, 2):
                for i in range(-1, 2):
                    g = np.array([i, j])
                    cell_id = ip + g
                    h = self._hash(cell_id)
                    o = 0.5 + 0.4 * np.sin(loop_time + h * 6.28318)
                    r = g[np.newaxis, np.newaxis, :] + o[..., np.newaxis] - fp
                    d = np.linalg.norm(r, axis=-1)
                    min_d = np.minimum(min_d, d)
            
            # Distorted grid
            grid_p = gp + 0.5 * np.stack([
                np.sin(gp[..., 1] * 2 + loop_time * 2),
                np.cos(gp[..., 0] * 2 + loop_time * 1.5)
            ], axis=-1)
            grid_val = np.abs(np.fmod(grid_p, 1.0) - 0.5)
            grid_line = 1 - np.clip(np.minimum(grid_val[..., 0], grid_val[..., 1]) / 0.1, 0, 1)
            
            height_val = fbm_val * 0.6 + min_d * 0.4
            result = height_val * 0.7 + grid_line * 0.3
            
        elif pattern == "fbm_noise":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            np_p = uv * 6.0 * scale
            result = self._fbm(np_p, loop_time)
            
        elif pattern == "cell_noise":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            cp = uv * 5.0 * scale
            ip = np.floor(cp).astype(int)
            fp = cp - np.floor(cp)
            
            f1 = np.ones((height, width))
            f2 = np.ones((height, width))
            
            for j in range(-1, 2):
                for i in range(-1, 2):
                    g = np.array([i, j])
                    cell_id = ip + g
                    h = self._hash(cell_id)
                    o = np.stack([
                        0.5 + 0.4 * np.sin(loop_time + h * 6.28318),
                        0.5 + 0.4 * np.cos(loop_time * 1.3 + h * 8.0)
                    ], axis=-1)
                    r = g[np.newaxis, np.newaxis, :] + o - fp
                    d = np.linalg.norm(r, axis=-1)
                    
                    mask = d < f1
                    f2 = np.where(mask, f1, np.minimum(f2, d))
                    f1 = np.where(mask, d, f1)
            
            result = f2 - f1
            
        elif pattern == "distorted_grid":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            dgp = (uv - 0.5) * 10.0 * scale
            
            distort = np.stack([
                np.sin(dgp[..., 1] * 0.5 + loop_time * 2) * 1.5 +
                np.sin(dgp[..., 1] * 2.0 - loop_time * 3) * 0.5,
                np.cos(dgp[..., 0] * 0.5 + loop_time * 1.7) * 1.5 +
                np.cos(dgp[..., 0] * 2.0 - loop_time * 2.5) * 0.5
            ], axis=-1)
            
            grid_p = dgp + distort
            grid_val = np.abs(np.fmod(grid_p, 1.0) - 0.5)
            lines = np.clip(np.minimum(grid_val[..., 0], grid_val[..., 1]) / 0.03, 0, 1)
            cell_shade = 0.3 + 0.7 * self._hash(np.floor(grid_p).astype(int))
            result = lines * cell_shade
            
        elif pattern == "height_map":
            loop_time = (time % 10.0) * 6.28318 / 10.0
            hp = uv * 4.0 * scale
            
            height_val = self._fbm(hp, loop_time)
            
            eps = 0.01
            h_l = self._fbm(hp - np.array([eps, 0]), loop_time)
            h_r = self._fbm(hp + np.array([eps, 0]), loop_time)
            h_u = self._fbm(hp + np.array([0, eps]), loop_time)
            h_d = self._fbm(hp - np.array([0, eps]), loop_time)
            
            normal = np.stack([h_l - h_r, h_d - h_u, np.ones_like(h_l) * 0.1], axis=-1)
            normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
            
            light_dir = np.array([np.cos(loop_time), np.sin(loop_time), 1.0])
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            lighting = 0.3 + 0.7 * np.maximum(0, np.sum(normal * light_dir, axis=-1))
            result = height_val * lighting
            
        elif pattern.startswith("rm_"):
            # Raymarched 3D shapes with camera controls
            result = self._raymarch_shape(width, height, pattern, time, scale, aspect,
                                          camera_dist, rot_x, rot_y)
            
        else:
            result = np.ones((height, width)) * 0.5
        
        return np.clip(result, 0, 1).astype(np.float32)

    def _raymarch_shape(self, width, height, shape, time, scale, aspect,
                        camera_dist=3.0, rot_x=0.0, rot_y=0.0):
        """Raymarch 3D shapes with camera controls."""
        loop_time = (time % 10.0) * 6.28318 / 10.0

        y, x = np.mgrid[0:height, 0:width]
        uv = np.stack([x / width, y / height], axis=-1)
        p = (uv - 0.5) * 2.0
        p[..., 0] *= aspect

        # Ray setup with controllable camera distance
        ro = np.array([0.0, 0.0, camera_dist])
        rd = np.stack([p[..., 0], p[..., 1], np.full((height, width), -1.5)], axis=-1)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)

        # Rotation from controls + time animation
        angle_x = np.radians(rot_x) + loop_time * 0.7
        angle_y = np.radians(rot_y) + loop_time
        ca, sa = np.cos(angle_y), np.sin(angle_y)
        cb, sb = np.cos(angle_x), np.sin(angle_x)
        
        result = np.zeros((height, width))
        t = np.zeros((height, width))
        hit = np.zeros((height, width), dtype=bool)
        
        for _ in range(64):
            pos = ro + rd * t[..., np.newaxis]
            
            # Rotate
            rp = pos.copy()
            rp_xz = rp[..., [0, 2]].copy()
            rp[..., 0] = rp_xz[..., 0] * ca - rp_xz[..., 1] * sa
            rp[..., 2] = rp_xz[..., 0] * sa + rp_xz[..., 1] * ca
            rp_yz = rp[..., [1, 2]].copy()
            rp[..., 1] = rp_yz[..., 0] * cb - rp_yz[..., 1] * sb
            rp[..., 2] = rp_yz[..., 0] * sb + rp_yz[..., 1] * cb
            
            # SDF based on shape
            if shape == "rm_cube":
                q = np.abs(rp) - 0.8
                d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0)
            elif shape == "rm_sphere":
                bump = 0.15 * np.sin(rp[..., 0] * 8 + loop_time * 2) * np.sin(rp[..., 1] * 8) * np.sin(rp[..., 2] * 8)
                d = np.linalg.norm(rp, axis=-1) - 1.0 - bump
            elif shape == "rm_torus":
                q = np.stack([np.linalg.norm(rp[..., [0, 2]], axis=-1) - 0.8, rp[..., 1]], axis=-1)
                d = np.linalg.norm(q, axis=-1) - 0.3
            elif shape == "rm_octahedron":
                ap = np.abs(rp)
                d = (ap[..., 0] + ap[..., 1] + ap[..., 2] - 1.2) * 0.577
            elif shape == "rm_gyroid":
                s = 3.0
                gyroid = np.sin(rp[..., 0] * s) * np.cos(rp[..., 1] * s) + \
                         np.sin(rp[..., 1] * s) * np.cos(rp[..., 2] * s) + \
                         np.sin(rp[..., 2] * s) * np.cos(rp[..., 0] * s)
                sphere = np.linalg.norm(rp, axis=-1) - 1.5
                d = np.maximum(np.abs(gyroid) - 0.1, sphere)
            elif shape == "rm_menger":
                d = np.linalg.norm(np.maximum(np.abs(rp) - 1.0, 0), axis=-1)
                s = 1.0
                for _ in range(3):
                    a = np.mod(rp * s, 2.0) - 1.0
                    s *= 3.0
                    r = np.abs(1.0 - 3.0 * np.abs(a))
                    c = np.maximum(r[..., 0], np.maximum(r[..., 1], r[..., 2]))
                    c = (c - 1.0) / s
                    d = np.maximum(d, c)
            else:
                d = np.linalg.norm(rp, axis=-1) - 1.0
            
            new_hit = (~hit) & (d < 0.001)
            hit |= new_hit
            
            t = np.where(hit | (t > 10), t, t + np.maximum(d, 0.001))
        
        # Lighting (simplified)
        result = np.where(hit, 1.0 - t * 0.08, 0.0)
        return np.clip(result, 0, 1).astype(np.float32)

    def generate(self, width, height, pattern, time, scale, seed, batch_size=1, loop_frames=0,
                 camera_distance=3.0, rotation_x=0.0, rotation_y=0.0):
        results = []

        if loop_frames > 0:
            # Generate animation frames
            for i in range(loop_frames):
                frame_time = (i / loop_frames) * 10.0  # 10 second loop
                frame = self._generate_pattern(width, height, pattern, frame_time, scale, seed,
                                               camera_distance, rotation_x, rotation_y)
                results.append(frame)
        else:
            # Generate batch with same time
            for b in range(batch_size):
                frame = self._generate_pattern(width, height, pattern, time, scale, seed + b,
                                               camera_distance, rotation_x, rotation_y)
                results.append(frame)

        # Stack and convert to RGB
        grey_stack = np.stack(results)
        rgb_stack = np.stack([grey_stack] * 3, axis=-1)

        image_tensor = torch.from_numpy(rgb_stack)
        mask_tensor = torch.from_numpy(grey_stack)

        return (image_tensor, mask_tensor)


class KoshiShapeMorph:
    """Morph between two patterns/shapes."""

    CATEGORY = "Koshi/Generators"
    FUNCTION = "morph"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["linear", "smooth", "ease_in", "ease_out", "sine"],),
            }
        }

    def morph(self, image_a, image_b, blend, blend_mode):
        # Apply easing
        if blend_mode == "smooth":
            t = blend * blend * (3 - 2 * blend)
        elif blend_mode == "ease_in":
            t = blend * blend
        elif blend_mode == "ease_out":
            t = 1 - (1 - blend) * (1 - blend)
        elif blend_mode == "sine":
            t = 0.5 - 0.5 * np.cos(blend * np.pi)
        else:
            t = blend
        
        # Morph
        result = image_a * (1 - t) + image_b * t
        
        # Create mask from luminance
        mask = result[..., 0] * 0.299 + result[..., 1] * 0.587 + result[..., 2] * 0.114
        
        return (result, mask)


class KoshiNoiseDisplace:
    """Apply noise displacement to an image or mask."""

    CATEGORY = "Koshi/Generators"
    FUNCTION = "displace"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
            }
        }

    def _fbm(self, p, seed, octaves):
        np.random.seed(seed)
        result = np.zeros_like(p[..., 0])
        amp = 0.5
        for i in range(octaves):
            noise = np.sin(p[..., 0] * 12.9898 + p[..., 1] * 78.233 + i) * 43758.5453
            noise = np.fmod(noise, 1.0)
            result += amp * noise
            p = p * 2.0
            amp *= 0.5
        return result

    def displace(self, image, strength, scale, octaves, seed, time=0.0):
        batch, height, width, channels = image.shape
        results = []
        
        for b in range(batch):
            img = image[b].cpu().numpy()
            
            # Create displacement field
            y, x = np.mgrid[0:height, 0:width]
            uv = np.stack([x / width, y / height], axis=-1) * scale
            
            # Animated noise
            uv_offset = np.array([time * 0.1, time * 0.07])
            
            dx = self._fbm(uv + uv_offset, seed, octaves)
            dy = self._fbm(uv + uv_offset + 100, seed + 1, octaves)
            
            # Apply displacement
            new_x = np.clip(x + dx * width * strength, 0, width - 1).astype(int)
            new_y = np.clip(y + dy * height * strength, 0, height - 1).astype(int)
            
            displaced = img[new_y, new_x]
            results.append(displaced)
        
        return (torch.from_numpy(np.stack(results)),)
