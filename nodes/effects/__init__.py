"""Koshi Effects Nodes - Unified effects + shader nodes."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .koshi_effects import KoshiEffects
    NODE_CLASS_MAPPINGS["Koshi_Effects"] = KoshiEffects
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Effects"] = "░▀░ Koshi Effects"
except ImportError:
    pass

try:
    from .bloom import BloomShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Bloom"] = BloomShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Bloom"] = "░▀░ KN Bloom"
except ImportError:
    pass

try:
    from .chromatic_aberration import KoshiChromaticAberration
    NODE_CLASS_MAPPINGS["Koshi_ChromaticAberration"] = KoshiChromaticAberration
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_ChromaticAberration"] = "░▀░ KN Chromatic Aberration"
except ImportError:
    pass

try:
    from .glitch import GlitchShaderNode
    NODE_CLASS_MAPPINGS["Koshi_Glitch"] = GlitchShaderNode
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Glitch"] = "░▀░ KN Glitch"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
