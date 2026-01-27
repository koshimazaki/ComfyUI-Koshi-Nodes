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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
