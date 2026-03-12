"""Koshi Export Nodes - OLED screen emulation and sprite sheets."""
from .oled_screen import KoshiOLEDScreen, KoshiSpriteSheet

NODE_CLASS_MAPPINGS = {
    "Koshi_OLEDScreen": KoshiOLEDScreen,
    "Koshi_SpriteSheet": KoshiSpriteSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_OLEDScreen": "░▒░ KN SIDKIT OLED",
    "Koshi_SpriteSheet": "░▒░ KN Sprite Sheet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
