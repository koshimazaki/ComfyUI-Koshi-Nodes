"""
ComfyUI-Koshi-Nodes
Comprehensive node pack for image processing, FLUX motion, effects, generators, and more.
"""

import importlib
import logging
import os

logger = logging.getLogger("Koshi")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CATEGORIES = [
    "nodes.image.dither",
    "nodes.image.greyscale",
    "nodes.image.binary",
    "nodes.effects",
    "nodes.export",
    "nodes.utility",
    "nodes.generators",
    "nodes.flux_motion",
    "nodes.audio",
]

def load_nodes():
    """Dynamically load all node modules."""
    base_path = os.path.dirname(__file__)

    for category in NODE_CATEGORIES:
        module_path = os.path.join(base_path, *category.split("."))

        if not os.path.exists(module_path):
            continue

        try:
            module = importlib.import_module(f".{category}", package=__name__)

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        except ImportError as e:
            logger.debug("Skipping %s: %s", category, e)
        except Exception as e:
            logger.warning("Error loading %s: %s", category, e)

load_nodes()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
