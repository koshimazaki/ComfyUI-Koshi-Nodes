"""Koshi Flux Motion Nodes - Animation and motion for FLUX models."""

import logging

logger = logging.getLogger("koshi.flux_motion")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .schedule import NODE_CLASS_MAPPINGS as schedule_nodes
    from .schedule import NODE_DISPLAY_NAME_MAPPINGS as schedule_names
    NODE_CLASS_MAPPINGS.update(schedule_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(schedule_names)
except ImportError as e:
    logger.debug(f"Failed to load schedule nodes: {e}")

try:
    from .motion_engine import NODE_CLASS_MAPPINGS as motion_nodes
    from .motion_engine import NODE_DISPLAY_NAME_MAPPINGS as motion_names
    NODE_CLASS_MAPPINGS.update(motion_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(motion_names)
except ImportError as e:
    logger.debug(f"Failed to load motion engine nodes: {e}")

try:
    from .feedback import NODE_CLASS_MAPPINGS as feedback_nodes
    from .feedback import NODE_DISPLAY_NAME_MAPPINGS as feedback_names
    NODE_CLASS_MAPPINGS.update(feedback_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(feedback_names)
except ImportError as e:
    logger.debug(f"Failed to load feedback nodes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
