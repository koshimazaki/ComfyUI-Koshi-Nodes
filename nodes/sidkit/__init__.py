"""SIDKIT Nodes - Export and preview for OLED displays and Teensy."""
from .export import SIDKITExport
from .oled_screen import SIDKITOLEDScreen, SIDKITSpriteSheet

NODE_CLASS_MAPPINGS = {
    "SIDKIT_Export": SIDKITExport,
    "SIDKIT_OLEDScreen": SIDKITOLEDScreen,
    "SIDKIT_SpriteSheet": SIDKITSpriteSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SIDKIT_Export": "░▒░ SIDKIT Export",
    "SIDKIT_OLEDScreen": "░▒░ SIDKIT OLED Screen",
    "SIDKIT_SpriteSheet": "░▒░ SIDKIT Sprite Sheet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
