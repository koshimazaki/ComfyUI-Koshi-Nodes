"""Post-processing effects for Glitch Candies."""
import numpy as np
from .utils import fbm


def noise_displace(image, strength=0.1, scale=10.0, octaves=4, seed=0, time=0.0):
    """Apply noise-based displacement to an image."""
    if strength == 0:
        return image

    np.random.seed(seed)
    height, width = image.shape[:2]

    # Create coordinate grids
    y, x = np.mgrid[0:height, 0:width]
    uv = np.stack([x / width, y / height], axis=-1) * scale

    # Animated noise offset
    uv_offset = np.array([time * 0.1, time * 0.07])

    # Generate displacement fields
    dx = _fbm_simple(uv + uv_offset, seed, octaves)
    dy = _fbm_simple(uv + uv_offset + 100, seed + 1, octaves)

    # Apply displacement
    new_x = np.clip(x + dx * width * strength, 0, width - 1).astype(int)
    new_y = np.clip(y + dy * height * strength, 0, height - 1).astype(int)

    if len(image.shape) == 3:
        return image[new_y, new_x]
    return image[new_y, new_x]


def _fbm_simple(p, seed, octaves):
    """Simple FBM for displacement."""
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


def noise_overlay(image, amount=0.0, scale=5.0, time=0.0):
    """Add noise overlay to image."""
    if amount == 0:
        return image

    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    uv = np.stack([x / width, y / height], axis=-1) * scale

    noise = _fbm_simple(uv, int(time * 100) % 10000, 4)
    noise = noise * 2 - 1  # -1 to 1

    if len(image.shape) == 3:
        return np.clip(image + noise[..., np.newaxis] * amount, 0, 1)
    return np.clip(image + noise * amount, 0, 1)


def blend_images(image_a, image_b, blend=0.5, mode="linear"):
    """Blend two images with various modes."""
    # Apply easing
    if mode == "smooth":
        t = blend * blend * (3 - 2 * blend)
    elif mode == "ease_in":
        t = blend * blend
    elif mode == "ease_out":
        t = 1 - (1 - blend) * (1 - blend)
    elif mode == "sine":
        t = 0.5 - 0.5 * np.cos(blend * np.pi)
    else:  # linear
        t = blend

    return image_a * (1 - t) + image_b * t


def apply_glitch_lines(image, intensity=0.1, time=0.0):
    """Add horizontal glitch lines."""
    if intensity == 0:
        return image

    height = image.shape[0]
    y_coords = np.arange(height)

    # Random line positions based on time
    np.random.seed(int(time * 100) % 10000)
    glitch_mask = np.random.rand(height) > (1 - intensity * 0.1)

    # Shift affected lines
    shift = (np.random.rand(height) * 20 - 10).astype(int)

    result = image.copy()
    for y in range(height):
        if glitch_mask[y]:
            result[y] = np.roll(image[y], shift[y], axis=0)

    return result


def apply_scanlines(image, intensity=0.1, frequency=2.0):
    """Add CRT-style scanlines."""
    if intensity == 0:
        return image

    height = image.shape[0]
    scanline = np.sin(np.arange(height) * frequency * np.pi)
    scanline = (scanline * 0.5 + 0.5) * intensity

    if len(image.shape) == 3:
        return np.clip(image - scanline[:, np.newaxis, np.newaxis], 0, 1)
    return np.clip(image - scanline[:, np.newaxis], 0, 1)


def apply_vignette(image, intensity=0.3):
    """Add vignette effect."""
    if intensity == 0:
        return image

    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    uv = np.stack([(x / width - 0.5) * 2, (y / height - 0.5) * 2], axis=-1)

    dist = np.linalg.norm(uv, axis=-1)
    vignette = 1 - np.clip(dist * intensity, 0, 1)

    if len(image.shape) == 3:
        return image * vignette[..., np.newaxis]
    return image * vignette


def colorize(grayscale, color_mode="grayscale"):
    """Convert grayscale to colored output."""
    if color_mode == "grayscale" or len(grayscale.shape) == 3:
        if len(grayscale.shape) == 2:
            return np.stack([grayscale] * 3, axis=-1)
        return grayscale

    colors = {
        "cyan": (0.0, 0.84, 0.99),
        "green": (0.0, 1.0, 0.4),
        "amber": (1.0, 0.75, 0.0),
        "purple": (0.6, 0.2, 1.0),
        "red": (1.0, 0.2, 0.1),
    }

    if color_mode in colors:
        c = colors[color_mode]
        return np.stack([grayscale * c[0], grayscale * c[1], grayscale * c[2]], axis=-1)

    return np.stack([grayscale] * 3, axis=-1)
