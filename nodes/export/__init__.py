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
    "Koshi_SIDKITScreen": "░▒░ KN SIDKIT Screen",
    "Koshi_OLEDPreview": "░▒░ KN OLED Preview",
    "Koshi_PixelScaler": "░▒░ KN Pixel Scaler",
    "Koshi_SpriteSheet": "░▒░ KN Sprite Sheet",
    "Koshi_OLEDScreen": "░▒░ KN OLED Screen",
    "Koshi_XBMExport": "░▒░ KN XBM Export",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
