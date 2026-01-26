"""Koshi Utility Nodes."""
from .metadata_node import KoshiCaptureSettings, KoshiSaveMetadata, KoshiDisplayMetadata

NODE_CLASS_MAPPINGS = {
    "Koshi_CaptureSettings": KoshiCaptureSettings,
    "Koshi_SaveMetadata": KoshiSaveMetadata,
    "Koshi_DisplayMetadata": KoshiDisplayMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_CaptureSettings": "Koshi Capture Settings",
    "Koshi_SaveMetadata": "Koshi Save Metadata",
    "Koshi_DisplayMetadata": "Koshi Display Metadata",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
