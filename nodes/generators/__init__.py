"""Koshi Generator Nodes - Unified procedural pattern generator."""
from .glitch_candies import (
    KoshiGlitchCandies,
    KoshiShapeMorph,
    KoshiNoiseDisplace,
)

NODE_CLASS_MAPPINGS = {
    "Koshi_GlitchCandies": KoshiGlitchCandies,
    "Koshi_ShapeMorph": KoshiShapeMorph,
    "Koshi_NoiseDisplace": KoshiNoiseDisplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_GlitchCandies": "▄█▄ Glitch Candies",
    "Koshi_ShapeMorph": "▄█▄ Shape Morph",
    "Koshi_NoiseDisplace": "▄█▄ Noise Displace",
}

# Import raymarcher from effects folder
try:
    import importlib.util, os, sys
    _ray_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "effects", "raymarcher.py")
    if os.path.exists(_ray_path):
        _spec = importlib.util.spec_from_file_location("koshi_raymarcher", _ray_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        DitheringRaymarcher = _mod.DitheringRaymarcher
        NODE_CLASS_MAPPINGS["Koshi_Raymarcher"] = DitheringRaymarcher
        NODE_DISPLAY_NAME_MAPPINGS["Koshi_Raymarcher"] = "▄█▄ Raymarcher"
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
