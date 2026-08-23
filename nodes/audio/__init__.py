"""Koshi Audio and conditioning nodes."""

import logging

from .akuspace import NODE_CLASS_MAPPINGS as _akuspace_nodes
from .akuspace import NODE_DISPLAY_NAME_MAPPINGS as _akuspace_names

logger = logging.getLogger("koshi.audio")

# Copy rather than alias so merging further nodes below never mutates the
# submodule's own mappings.
NODE_CLASS_MAPPINGS = dict(_akuspace_nodes)
NODE_DISPLAY_NAME_MAPPINGS = dict(_akuspace_names)

try:
    from .audio_motion_schedule import NODE_CLASS_MAPPINGS as _ams_nodes
    from .audio_motion_schedule import NODE_DISPLAY_NAME_MAPPINGS as _ams_names
    NODE_CLASS_MAPPINGS.update(_ams_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ams_names)
except ImportError as exc:
    logger.debug("Failed to load audio motion schedule node: %s", exc)

try:
    from .aligned_ref import NODE_CLASS_MAPPINGS as _aligned_nodes
    from .aligned_ref import NODE_DISPLAY_NAME_MAPPINGS as _aligned_names
    NODE_CLASS_MAPPINGS.update(_aligned_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_aligned_names)
except ImportError as exc:
    logger.debug("Failed to load AKUSPACE aligned reference node: %s", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
