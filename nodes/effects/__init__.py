"""Koshi Effects Nodes - Bloom, Glitch, Chromatic Aberration, Hologram, Raymarcher."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import bloom
try:
    from .bloom import BloomShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Bloom"] = BloomShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Bloom"] = "▄▀▄ KN Bloom"
except ImportError:
    pass

# Import chromatic aberration (alien.js)
try:
    from .chromatic_aberration import KoshiChromaticAberration
    NODE_CLASS_MAPPINGS["Koshi_ChromaticAberration"] = KoshiChromaticAberration
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_ChromaticAberration"] = "▄▀▄ KN Chromatic Aberration"
except ImportError:
    pass

# Import hologram effects (alien.js + CreaturesSite)
try:
    from .hologram import KoshiHologram, KoshiScanlines, KoshiVideoGlitch
    NODE_CLASS_MAPPINGS["Koshi_Hologram"] = KoshiHologram
    NODE_CLASS_MAPPINGS["Koshi_Scanlines"] = KoshiScanlines
    NODE_CLASS_MAPPINGS["Koshi_VideoGlitch"] = KoshiVideoGlitch
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Hologram"] = "▄▀▄ KN Hologram"
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Scanlines"] = "▄▀▄ KN Scanlines"
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_VideoGlitch"] = "▄▀▄ KN Video Glitch"
except ImportError:
    pass

# Import glitch
try:
    from .glitch import GlitchShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Glitch"] = GlitchShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Glitch"] = "▄▀▄ KN Glitch"
except ImportError:
    pass

# Import raymarcher
try:
    from .raymarcher import DitheringRaymarcher
    NODE_CLASS_MAPPINGS["Koshi_Raymarcher"] = DitheringRaymarcher
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Raymarcher"] = "▄▀▄ KN Raymarcher"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
