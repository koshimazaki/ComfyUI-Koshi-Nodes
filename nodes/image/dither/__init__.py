"""Koshi Dither Node - SIDKIT Edition."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Import advanced filter if available
try:
    from .filter import ImageDitheringFilter
    NODE_CLASS_MAPPINGS["Koshi_DitheringFilter"] = ImageDitheringFilter
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_DitheringFilter"] = "Koshi Dithering Filter (GPU)"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
