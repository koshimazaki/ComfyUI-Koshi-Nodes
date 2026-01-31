"""Shared utilities for Glitch Candies generators."""
import numpy as np
import os
import uuid

try:
    from PIL import Image
    import folder_paths
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False


def save_preview(image_tensor, prefix="koshi"):
    """Save images for ComfyUI preview."""
    if not PREVIEW_AVAILABLE:
        return []

    results = []
    output_dir = folder_paths.get_temp_directory()

    batch = image_tensor if len(image_tensor.shape) == 4 else image_tensor.unsqueeze(0)
    for i in range(batch.shape[0]):
        img_np = (np.clip(batch[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{i}.png"
        Image.fromarray(img_np).save(os.path.join(output_dir, filename))
        results.append({"filename": filename, "subfolder": "", "type": "temp"})

    return results


def hash2d(p):
    """2D hash function."""
    return np.fmod(np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7) * 43758.5453, 1.0)


def hash3d(p):
    """3D hash function."""
    return np.fmod(np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7 + p[..., 2] * 74.7) * 43758.5453, 1.0)


def value_noise(p, t):
    """Value noise with time animation."""
    i = np.floor(p).astype(int)
    f = p - np.floor(p)
    f = f * f * (3.0 - 2.0 * f)

    def h(offset):
        ip = np.stack([i[..., 0] + offset[0], i[..., 1] + offset[1]], axis=-1)
        hv = hash2d(ip)
        return hv + 0.2 * np.sin(t + hv * 6.28318)

    a = h([0, 0])
    b = h([1, 0])
    c = h([0, 1])
    d = h([1, 1])

    return (a * (1 - f[..., 0]) + b * f[..., 0]) * (1 - f[..., 1]) + \
           (c * (1 - f[..., 0]) + d * f[..., 0]) * f[..., 1]


def fbm(p, t, octaves=5):
    """Fractal Brownian Motion."""
    result = np.zeros_like(p[..., 0])
    amp = 0.5
    for i in range(octaves):
        result += amp * value_noise(p, t + i * 0.5)
        p = p * 2.0
        amp *= 0.5
    return result


def smoothstep(edge0, edge1, x):
    """GLSL smoothstep function."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_uv(width, height, scale=1.0, center=True):
    """Create UV coordinate grid."""
    y, x = np.mgrid[0:height, 0:width]
    uv = np.stack([x / width, y / height], axis=-1)
    if center:
        uv = (uv - 0.5) * 2.0 * scale
    else:
        uv = uv * scale
    return uv


def rot_x(a):
    """3D rotation matrix around X axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    """3D rotation matrix around Y axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
