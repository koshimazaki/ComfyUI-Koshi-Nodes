"""Koshi Utility Nodes."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .koshi_metadata import KoshiMetadata
    NODE_CLASS_MAPPINGS["Koshi_Metadata"] = KoshiMetadata
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Metadata"] = "◊ Koshi Metadata"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
