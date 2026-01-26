"""Koshi Export Nodes - OLED screen emulation and SIDKIT export."""
from .sidkit import SIDKITExport
from .oled_preview import KoshiOLEDPreview
from .oled_screen import (
    KoshiPixelScaler,
    KoshiSpriteSheet,
    KoshiOLEDScreen,
    KoshiXBMExport,
)

NODE_CLASS_MAPPINGS = {
    # SIDKIT export (renamed to Screen)
    "Koshi_SIDKITScreen": SIDKITExport,
    # Legacy OLED preview (keep for compatibility)
    "Koshi_OLEDPreview": KoshiOLEDPreview,
    # New enhanced nodes
    "Koshi_PixelScaler": KoshiPixelScaler,
    "Koshi_SpriteSheet": KoshiSpriteSheet,
    "Koshi_OLEDScreen": KoshiOLEDScreen,
    "Koshi_XBMExport": KoshiXBMExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_SIDKITScreen": "SIDKIT Screen",
    "Koshi_OLEDPreview": "Koshi OLED Preview (Legacy)",
    "Koshi_PixelScaler": "Koshi Pixel Scaler",
    "Koshi_SpriteSheet": "Koshi Sprite Sheet",
    "Koshi_OLEDScreen": "Koshi OLED Screen",
    "Koshi_XBMExport": "Koshi XBM Export",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
