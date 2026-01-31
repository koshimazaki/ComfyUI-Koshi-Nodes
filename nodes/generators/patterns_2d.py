"""2D pattern generators for Glitch Candies."""
import numpy as np
from .utils import hash2d, fbm, value_noise, make_uv


def waves(uv, time):
    """Animated wave interference pattern."""
    result = np.sin(uv[..., 0] * 20 + time) * np.sin(uv[..., 1] * 20 + time * 0.7)
    result += np.sin(np.linalg.norm(uv - 0.5, axis=-1) * 30 - time * 2) * 0.5
    return result * 0.5 + 0.5


def circles(uv, time):
    """Concentric circles emanating from center."""
    d = np.linalg.norm(uv - 0.5, axis=-1)
    return np.sin(d * 40 - time * 3) * 0.5 + 0.5


def plasma(uv, time):
    """Classic plasma effect."""
    v1 = np.sin(uv[..., 0] * 10 + time)
    v2 = np.sin(uv[..., 1] * 10 + time * 1.2)
    v3 = np.sin((uv[..., 0] + uv[..., 1]) * 10 + time * 0.8)
    v4 = np.sin(np.linalg.norm(uv - 0.5, axis=-1) * 20 + time * 1.5)
    return (v1 + v2 + v3 + v4) * 0.25 + 0.5


def voronoi(uv, time, scale):
    """Animated voronoi cells."""
    loop_time = np.fmod(time, 10.0) * 0.628318
    p = uv * 5.0 * scale
    ip = np.floor(p).astype(int)
    fp = p - np.floor(p)

    min_d = np.full_like(uv[..., 0], 8.0)

    for j in range(-1, 2):
        for i in range(-1, 2):
            g = np.array([i, j])
            cell_id = np.stack([ip[..., 0] + i, ip[..., 1] + j], axis=-1)
            h = hash2d(cell_id)
            o = 0.5 + 0.4 * np.sin(loop_time + h * 6.28318)
            r = g + o - fp
            d = np.linalg.norm(r, axis=-1)
            min_d = np.minimum(min_d, d)

    return min_d


def checkerboard(uv, time, scale):
    """Animated checkerboard pattern."""
    p = uv * 8 * scale
    checker = np.floor(p[..., 0]) + np.floor(p[..., 1])
    checker = np.fmod(checker, 2)
    pulse = np.sin(time * 2) * 0.1
    return np.abs(checker) * (0.9 + pulse) + (1 - np.abs(checker)) * (0.1 - pulse)


def mandelbrot(uv, time, scale):
    """Mandelbrot fractal with zoom animation."""
    zoom = 2.0 + np.sin(time * 0.1) * 0.5
    center = np.array([-0.5, 0.0])
    c = (uv - 0.5) * zoom * scale + center

    z = np.zeros_like(c)
    result = np.zeros_like(c[..., 0])

    for i in range(50):
        mask = np.linalg.norm(z, axis=-1) < 2
        z_new = np.stack([
            z[..., 0] ** 2 - z[..., 1] ** 2 + c[..., 0],
            2 * z[..., 0] * z[..., 1] + c[..., 1]
        ], axis=-1)
        z = np.where(mask[..., np.newaxis], z_new, z)
        result = np.where(mask, result + 1, result)

    return result / 50.0


def julia(uv, time, scale):
    """Julia set with animated parameter."""
    c_val = np.array([
        -0.7 + np.sin(time * 0.3) * 0.2,
        0.27 + np.cos(time * 0.4) * 0.15
    ])
    z = (uv - 0.5) * 3.0 * scale
    result = np.zeros_like(z[..., 0])

    for i in range(50):
        mask = np.linalg.norm(z, axis=-1) < 2
        z_new = np.stack([
            z[..., 0] ** 2 - z[..., 1] ** 2 + c_val[0],
            2 * z[..., 0] * z[..., 1] + c_val[1]
        ], axis=-1)
        z = np.where(mask[..., np.newaxis], z_new, z)
        result = np.where(mask, result + 1, result)

    return result / 50.0


def sierpinski(uv, time, scale):
    """Sierpinski triangle pattern."""
    p = (uv - 0.5) * scale * 4
    p = p + np.array([0, -0.5])

    result = np.zeros_like(p[..., 0])
    for _ in range(8):
        p = np.abs(p)
        p = p - np.array([1, 1])
        p = p * 2
        result += np.where((p[..., 0] + p[..., 1]) < 0, 1, 0)

    return np.fmod(result + time * 0.5, 8) / 8


def swirl(uv, time, scale):
    """Swirling spiral pattern."""
    p = (uv - 0.5) * 2
    r = np.linalg.norm(p, axis=-1)
    a = np.arctan2(p[..., 1], p[..., 0])
    spiral = np.sin(a * 5 + r * 10 * scale - time * 2)
    return spiral * 0.5 + 0.5


def ripple(uv, time, scale):
    """Water ripple effect."""
    center1 = np.array([0.3, 0.3])
    center2 = np.array([0.7, 0.7])
    d1 = np.linalg.norm(uv - center1, axis=-1)
    d2 = np.linalg.norm(uv - center2, axis=-1)
    wave1 = np.sin(d1 * 30 * scale - time * 4)
    wave2 = np.sin(d2 * 25 * scale - time * 3.5)
    return (wave1 + wave2) * 0.25 + 0.5


def glitch_candies(uv, time, scale):
    """Main glitch candies pattern - FBM + voronoi blend."""
    loop_time = np.fmod(time, 10.0) * 0.628318
    gp = (uv - 0.5) * 8.0 * scale

    # FBM layer
    fbm_val = fbm(gp, loop_time)

    # Voronoi layer
    ip = np.floor(gp).astype(int)
    fp = gp - np.floor(gp)
    min_d = np.ones_like(gp[..., 0])

    for j in range(-1, 2):
        for i in range(-1, 2):
            cell_id = np.stack([ip[..., 0] + i, ip[..., 1] + j], axis=-1)
            h = hash2d(cell_id)
            o = 0.5 + 0.4 * np.sin(loop_time + h * 6.28318)
            r = np.array([i, j]) + o - fp
            d = np.linalg.norm(r, axis=-1)
            min_d = np.minimum(min_d, d)

    # Blend with animated weight
    blend = np.sin(loop_time) * 0.5 + 0.5
    result = fbm_val * blend + min_d * (1 - blend)

    # Add glitch lines
    glitch_y = np.sin(gp[..., 1] * 50 + loop_time * 10)
    glitch_mask = glitch_y > 0.95
    result = np.where(glitch_mask, 1 - result, result)

    return result


def fbm_noise(uv, time, scale):
    """Pure FBM noise pattern."""
    p = (uv - 0.5) * 8.0 * scale
    return fbm(p, time)


def cell_noise(uv, time, scale):
    """Cellular/Worley noise."""
    p = uv * 6.0 * scale
    ip = np.floor(p).astype(int)
    fp = p - np.floor(p)

    min_d1 = np.full_like(uv[..., 0], 8.0)
    min_d2 = np.full_like(uv[..., 0], 8.0)

    for j in range(-1, 2):
        for i in range(-1, 2):
            cell_id = np.stack([ip[..., 0] + i, ip[..., 1] + j], axis=-1)
            h = hash2d(cell_id)
            o = 0.5 + 0.4 * np.sin(time + h * 6.28318)
            r = np.array([i, j]) + o - fp
            d = np.linalg.norm(r, axis=-1)
            new_min1 = np.minimum(min_d1, d)
            new_min2 = np.where(d < min_d1, min_d1, np.minimum(min_d2, d))
            min_d1, min_d2 = new_min1, new_min2

    return min_d2 - min_d1


def distorted_grid(uv, time, scale):
    """Grid with noise distortion."""
    p = uv * 10 * scale
    offset = fbm(uv * 3, time) * 0.5
    p = p + offset
    grid = np.abs(np.sin(p[..., 0] * np.pi)) * np.abs(np.sin(p[..., 1] * np.pi))
    return grid


def height_map(uv, time, scale):
    """Terrain-like height map."""
    p = uv * 4 * scale
    h = fbm(p, time * 0.2, octaves=6)
    # Add ridges
    h = np.abs(h * 2 - 1)
    return h


def glitch_cubes(uv, time, scale):
    """Glitchy cube pattern."""
    p = uv * 8 * scale
    grid = np.floor(p)
    local = p - grid

    # Random offset per cell
    h = hash2d(grid)
    offset = np.sin(time * 3 + h * 6.28) * 0.3

    # Cube edges
    edge_x = np.abs(local[..., 0] - 0.5 + offset)
    edge_y = np.abs(local[..., 1] - 0.5 + offset)
    cube = np.maximum(edge_x, edge_y)

    # Glitch some cells
    glitch = h > 0.8
    result = np.where(glitch, 1 - cube, cube)

    return result


# Pattern registry
PATTERNS_2D = {
    "waves": waves,
    "circles": circles,
    "plasma": plasma,
    "voronoi": voronoi,
    "checkerboard": checkerboard,
    "mandelbrot": mandelbrot,
    "julia": julia,
    "sierpinski": sierpinski,
    "swirl": swirl,
    "ripple": ripple,
    "glitch_candies": glitch_candies,
    "fbm_noise": fbm_noise,
    "cell_noise": cell_noise,
    "distorted_grid": distorted_grid,
    "height_map": height_map,
    "glitch_cubes": glitch_cubes,
}


def generate_2d(pattern, width, height, time, scale, seed):
    """Generate a 2D pattern by name."""
    np.random.seed(seed)
    uv = make_uv(width, height, scale=1.0, center=False)

    if pattern in PATTERNS_2D:
        func = PATTERNS_2D[pattern]
        # Check function signature for scale parameter
        import inspect
        sig = inspect.signature(func)
        if 'scale' in sig.parameters:
            return func(uv, time, scale)
        else:
            return func(uv * scale, time)

    return np.zeros((height, width))
