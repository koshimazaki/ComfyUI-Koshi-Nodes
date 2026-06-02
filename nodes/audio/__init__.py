"""Koshi Audio Nodes - Audio reactive and injection."""

import logging

logger = logging.getLogger("koshi.audio")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .audio_motion_schedule import NODE_CLASS_MAPPINGS as _ams_nodes
    from .audio_motion_schedule import NODE_DISPLAY_NAME_MAPPINGS as _ams_names
    NODE_CLASS_MAPPINGS.update(_ams_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ams_names)
except ImportError as exc:
    logger.debug("Failed to load audio motion schedule node: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
