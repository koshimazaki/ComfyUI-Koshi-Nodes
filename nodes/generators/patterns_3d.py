"""3D raymarched pattern generators for Glitch Candies."""
import numpy as np
from .utils import rot_x, rot_y, fbm, make_uv


# =============================================================================
# SDF PRIMITIVES
# =============================================================================

def sdf_cube(p, size=0.8):
    """Signed distance to a cube."""
    q = np.abs(p) - size
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + \
           np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0)


def sdf_sphere(p, radius=1.0):
    """Signed distance to a sphere."""
    return np.linalg.norm(p, axis=-1) - radius


def sdf_torus(p, r1=0.7, r2=0.3):
    """Signed distance to a torus."""
    q = np.stack([np.linalg.norm(p[..., :2], axis=-1) - r1, p[..., 2]], axis=-1)
    return np.linalg.norm(q, axis=-1) - r2


def sdf_octahedron(p, s=1.0):
    """Signed distance to an octahedron."""
    p = np.abs(p)
    return (p[..., 0] + p[..., 1] + p[..., 2] - s) * 0.57735027


def sdf_gyroid(p, scale=3.0, thickness=0.03):
    """Signed distance to a gyroid surface."""
    ps = p * scale
    g = np.sin(ps[..., 0]) * np.cos(ps[..., 1]) + \
        np.sin(ps[..., 1]) * np.cos(ps[..., 2]) + \
        np.sin(ps[..., 2]) * np.cos(ps[..., 0])
    return np.abs(g) - thickness


def sdf_menger(p, iterations=3):
    """Signed distance to a Menger sponge."""
    d = sdf_cube(p, 1.0)
    s = 1.0

    for _ in range(iterations):
        a = np.fmod(np.abs(p) * s, 2.0) - 1.0
        s *= 3.0
        r = np.abs(1.0 - 3.0 * np.abs(a))

        da = np.maximum(r[..., 0], r[..., 1])
        db = np.maximum(r[..., 1], r[..., 2])
        dc = np.maximum(r[..., 0], r[..., 2])
        c = (np.minimum(np.minimum(da, db), dc) - 1.0) / s

        d = np.maximum(d, c)

    return d


def sdf_cylinder(p, h=1.0, r=0.5):
    """Signed distance to a cylinder."""
    d = np.abs(np.stack([np.linalg.norm(p[..., :2], axis=-1), p[..., 2]], axis=-1)) - np.array([r, h])
    return np.minimum(np.maximum(d[..., 0], d[..., 1]), 0.0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


def sdf_cone(p, angle=0.5, h=1.0):
    """Signed distance to a cone."""
    c = np.array([np.sin(angle), np.cos(angle)])
    q = np.stack([np.linalg.norm(p[..., :2], axis=-1), -p[..., 2]], axis=-1)
    d = np.linalg.norm(q - c * np.clip(np.sum(q * c, axis=-1, keepdims=True), 0, h), axis=-1)
    return d * np.where(q[..., 0] * c[1] - q[..., 1] * c[0] < 0, -1, 1)


def sdf_capsule(p, h=0.5, r=0.25):
    """Signed distance to a capsule."""
    p_clamped = np.stack([p[..., 0], np.clip(p[..., 1], -h, h), p[..., 2]], axis=-1)
    return np.linalg.norm(p - np.stack([np.zeros_like(p[..., 0]), p_clamped[..., 1], np.zeros_like(p[..., 2])], axis=-1), axis=-1) - r


def sdf_pyramid(p, h=1.0):
    """Signed distance to a pyramid."""
    m2 = h * h + 0.25
    p_abs = np.abs(p[..., :2])
    p_swapped = np.where(p_abs[..., 1:2] > p_abs[..., 0:1],
                         np.stack([p_abs[..., 1], p_abs[..., 0]], axis=-1),
                         p_abs)
    px, py = p_swapped[..., 0], p_swapped[..., 1]
    pz = p[..., 2]

    px -= 0.5
    py -= 0.5

    qx = pz
    qy = h * py - 0.5 * px
    qz = h * px + 0.5 * py

    s = np.maximum(-qx, 0.0)
    t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0)

    a = m2 * (qx + s) ** 2 + qy ** 2
    b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2

    d2 = np.where(np.minimum(qy, -qx * m2 - qy * 0.5) > 0, 0.0, np.minimum(a, b))

    return np.sqrt((d2 + qz ** 2) / m2) * np.sign(np.maximum(qz, -pz - 0.5))


# SDF registry
SDF_SHAPES = {
    "cube": sdf_cube,
    "sphere": sdf_sphere,
    "torus": sdf_torus,
    "octahedron": sdf_octahedron,
    "gyroid": sdf_gyroid,
    "menger": sdf_menger,
    "cylinder": sdf_cylinder,
    "cone": sdf_cone,
    "capsule": sdf_capsule,
    "pyramid": sdf_pyramid,
}


# =============================================================================
# RAYMARCHING
# =============================================================================

def raymarch(width, height, sdf_func, cam_dist=3.0, rot_x_deg=0.0, rot_y_deg=0.0,
             noise_disp=0.0, noise_freq=3.0, time=0.0):
    """Raymarch a scene and return grayscale result."""
    # Camera setup
    uv = make_uv(width, height, scale=1.0, center=True)
    aspect = width / height
    uv[..., 0] *= aspect

    # Ray direction
    rd = np.zeros((height, width, 3))
    rd[..., 0] = uv[..., 0]
    rd[..., 1] = uv[..., 1]
    rd[..., 2] = -1.5
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)

    # Apply rotation
    rx = rot_x(np.radians(rot_x_deg))
    ry = rot_y(np.radians(rot_y_deg))
    rot = rx @ ry
    rd = np.einsum('ij,...j->...i', rot, rd)

    # Ray origin
    ro = np.array([0, 0, cam_dist])
    ro = rot @ ro

    # Raymarch
    t = np.zeros((height, width))
    result = np.zeros((height, width))

    for _ in range(64):
        p = ro + rd * t[..., np.newaxis]

        # Apply noise displacement
        if noise_disp > 0:
            noise = fbm(p[..., :2] * noise_freq, time)
            p = p + rd * noise[..., np.newaxis] * noise_disp * 0.5

        d = sdf_func(p)
        t += d * 0.5

        # Hit detection
        hit = d < 0.001
        result = np.where(hit & (result == 0), 1.0 - t * 0.1, result)

        # Early exit for far rays
        if np.all(t > 10):
            break

    return np.clip(result, 0, 1)


def morph_shapes(width, height, shape_a, shape_b, morph=0.5,
                 cam_dist=3.0, rot_x_deg=0.0, rot_y_deg=0.0,
                 noise_disp=0.0, noise_freq=3.0, time=0.0):
    """Morph between two SDF shapes."""
    sdf_a = SDF_SHAPES.get(shape_a, sdf_sphere)
    sdf_b = SDF_SHAPES.get(shape_b, sdf_cube)

    def morphed_sdf(p):
        da = sdf_a(p)
        db = sdf_b(p)
        # Smooth blend
        t = morph * morph * (3 - 2 * morph)  # smoothstep
        return da * (1 - t) + db * t

    return raymarch(width, height, morphed_sdf, cam_dist, rot_x_deg, rot_y_deg,
                    noise_disp, noise_freq, time)


def generate_3d(pattern, width, height, time, scale, seed,
                cam_dist=3.0, rot_x_deg=0.0, rot_y_deg=0.0,
                shape_a="sphere", shape_b="cube", morph=0.0,
                noise_disp=0.0, noise_freq=3.0):
    """Generate a 3D raymarched pattern."""
    np.random.seed(seed)

    # Extract shape name from pattern like "rm_cube" -> "cube"
    if pattern.startswith("rm_"):
        shape_name = pattern[3:]
        if shape_name in SDF_SHAPES:
            return raymarch(width, height, SDF_SHAPES[shape_name],
                           cam_dist, rot_x_deg, rot_y_deg,
                           noise_disp, noise_freq, time)

    # Shape morph mode
    if pattern == "shape_morph":
        return morph_shapes(width, height, shape_a, shape_b, morph,
                           cam_dist, rot_x_deg, rot_y_deg,
                           noise_disp, noise_freq, time)

    return np.zeros((height, width))


# List of 3D patterns
PATTERNS_3D = ["rm_" + name for name in SDF_SHAPES.keys()] + ["shape_morph"]
