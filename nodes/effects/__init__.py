"""Koshi Effects Nodes - Unified effects + individual effect nodes."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Unified Effects node (main)
try:
    from .koshi_effects import KoshiEffects
    NODE_CLASS_MAPPINGS["Koshi_Effects"] = KoshiEffects
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Effects"] = "░▀░ Koshi Effects"
except ImportError:
    pass

# Import bloom
try:
    from .bloom import BloomShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Bloom"] = BloomShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Bloom"] = "░▀░ KN Bloom"
except ImportError:
    pass

# Import chromatic aberration (alien.js)
try:
    from .chromatic_aberration import KoshiChromaticAberration
    NODE_CLASS_MAPPINGS["Koshi_ChromaticAberration"] = KoshiChromaticAberration
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_ChromaticAberration"] = "░▀░ KN Chromatic Aberration"
except ImportError:
    pass

# Import hologram effects (alien.js + CreaturesSite)
try:
    from .hologram import KoshiHologram, KoshiScanlines, KoshiVideoGlitch
    NODE_CLASS_MAPPINGS["Koshi_Hologram"] = KoshiHologram
    NODE_CLASS_MAPPINGS["Koshi_Scanlines"] = KoshiScanlines
    NODE_CLASS_MAPPINGS["Koshi_VideoGlitch"] = KoshiVideoGlitch
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Hologram"] = "░▀░ KN Hologram"
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Scanlines"] = "░▀░ KN Scanlines"
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_VideoGlitch"] = "░▀░ KN Video Glitch"
except ImportError:
    pass

# Import glitch
try:
    from .glitch import GlitchShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Glitch"] = GlitchShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Glitch"] = "░▀░ KN Glitch"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
