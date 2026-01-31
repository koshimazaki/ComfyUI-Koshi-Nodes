"""
ComfyUI-Koshi-Nodes
Comprehensive node pack for image processing, FLUX motion, effects, generators, and more.
"""

import importlib.util
import logging
import os
import sys

logger = logging.getLogger("Koshi")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CATEGORIES = [
    "nodes.sidkit",
    "nodes.effects",
    "nodes.generators",
    "nodes.utility",
    "nodes.flux_motion",
    # Legacy - to be removed after migration
    "nodes.image.dither",
    "nodes.image.greyscale",
    "nodes.image.binary",
    "nodes.export",
    "nodes.audio",
]


def load_nodes():
    """Dynamically load all node modules using file paths to avoid conflicts with ComfyUI's nodes module."""
    base_path = os.path.dirname(__file__)

    for category in NODE_CATEGORIES:
        # Convert category to file path
        module_rel_path = category.replace(".", os.sep)
        module_path = os.path.join(base_path, module_rel_path, "__init__.py")

        if not os.path.exists(module_path):
            module_path = os.path.join(base_path, module_rel_path + ".py")
            if not os.path.exists(module_path):
                continue

        try:
            # Use spec_from_file_location to avoid conflicts with ComfyUI's nodes module
            # This loads directly from file path instead of relying on Python's import system
            module_name = f"koshi_{category.replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        except Exception as e:
            logger.warning("Error loading %s: %s", category, e)


load_nodes()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
