"""Koshi Utility Nodes."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Unified Metadata node (recommended)
try:
    from .koshi_metadata import KoshiMetadata
    NODE_CLASS_MAPPINGS["Koshi_Metadata"] = KoshiMetadata
    NODE_DISPLAY_NAME_MAPPINGS["Koshi_Metadata"] = "◊ Koshi Metadata"
except ImportError:
    pass

# Legacy nodes (kept for compatibility)
try:
    from .metadata_node import KoshiCaptureSettings, KoshiSaveMetadata, KoshiDisplayMetadata
    NODE_CLASS_MAPPINGS.update({
        "Koshi_CaptureSettings": KoshiCaptureSettings,
        "Koshi_SaveMetadata": KoshiSaveMetadata,
        "Koshi_DisplayMetadata": KoshiDisplayMetadata,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "Koshi_CaptureSettings": "◊ KN Capture Settings [Legacy]",
        "Koshi_SaveMetadata": "◊ KN Save Metadata [Legacy]",
        "Koshi_DisplayMetadata": "◊ KN Display Metadata [Legacy]",
    })
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
