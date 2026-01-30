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
    import sys
    base_path = os.path.dirname(__file__)

    # Ensure base path is in sys.path for imports
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    for category in NODE_CATEGORIES:
        module_path = os.path.join(base_path, *category.split("."))

        if not os.path.exists(module_path):
            continue

        try:
            # Use absolute import instead of relative
            module = importlib.import_module(category)

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
