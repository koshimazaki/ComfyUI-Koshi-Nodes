"""Koshi Generator Nodes - Procedural patterns and 3D shapes."""
from .glitch_candies import KoshiGlitchCandies, KoshiShapeMorph, KoshiNoiseDisplace

NODE_CLASS_MAPPINGS = {
    "Koshi_GlitchCandies": KoshiGlitchCandies,
    "Koshi_ShapeMorph": KoshiShapeMorph,
    "Koshi_NoiseDisplace": KoshiNoiseDisplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_GlitchCandies": "▄█▄ KN Glitch Candies",
    "Koshi_ShapeMorph": "▄█▄ KN Shape Morph",
    "Koshi_NoiseDisplace": "▄█▄ KN Noise Displace",
}

# Import raymarcher from effects folder (generates content)
try:
    from ..effects.raymarcher import DitheringRaymarcher
    NODE_CLASS_MAPPINGS["Koshi_Raymarcher"] = DitheringRaymarcher
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Raymarcher"] = "▄█▄ KN Raymarcher"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
