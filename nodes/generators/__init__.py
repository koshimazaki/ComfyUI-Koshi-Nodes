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
    from ..effects.raymarcher import DitheringRaymarcher
    NODE_CLASS_MAPPINGS["Koshi_Raymarcher"] = DitheringRaymarcher
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Raymarcher"] = "▄█▄ Raymarcher"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
