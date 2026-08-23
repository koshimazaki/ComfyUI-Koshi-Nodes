"""Sprint 2: Verify all node classes register correctly with proper ComfyUI interface."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build expected node keys programmatically
_PFX = "Koshi_"
_SUFFIXES = [
    # Image processing
    "Dither", "DitheringFilter", "Greyscale", "Binary",
    # Effects
    "Effects", "Bloom", "ChromaticAberration", "Glitch",
    # Export / SIDKIT
    "OLEDScreen", "SpriteSheet",
    # Generators
    "GlitchCandies", "ShapeMorph", "NoiseDisplace", "Raymarcher",
    # Utility
    "Metadata",
    # Space / Conditioning
    "AKUSPACEPrompt", "AKUSPACETextEncode",
    # Audio
    "AudioMotionSchedule",
    # Flux Motion
    "Schedule", "MotionEngine", "Feedback",
]
# Deliberately unprefixed: the id has to match the standalone copy shipped in
# the AKUSPACE session kit, so a graph built against either resolves in both.
_UNPREFIXED = {"AKUSPACEReferenceAudioAligned"}
EXPECTED = {_PFX + s for s in _SUFFIXES} | _UNPREFIXED


def _gather_class_map():
    """Collect NODE_CLASS_MAPPINGS from all categories."""
    from nodes.image.dither import NODE_CLASS_MAPPINGS as dither
    from nodes.image.greyscale import NODE_CLASS_MAPPINGS as greyscale
    from nodes.image.binary import NODE_CLASS_MAPPINGS as binary
    from nodes.effects import NODE_CLASS_MAPPINGS as effects
    from nodes.export import NODE_CLASS_MAPPINGS as exp
    from nodes.generators import NODE_CLASS_MAPPINGS as gen
    from nodes.utility import NODE_CLASS_MAPPINGS as util
    from nodes.audio import NODE_CLASS_MAPPINGS as audio
    from nodes.flux_motion import NODE_CLASS_MAPPINGS as fm

    combined = {}
    for m in [dither, greyscale, binary, effects, exp, gen, util, audio, fm]:
        combined.update(m)
    return combined


def _gather_name_map():
    """Collect NODE_DISPLAY_NAME_MAPPINGS from all categories."""
    from nodes.image.dither import NODE_DISPLAY_NAME_MAPPINGS as dither
    from nodes.image.greyscale import NODE_DISPLAY_NAME_MAPPINGS as greyscale
    from nodes.image.binary import NODE_DISPLAY_NAME_MAPPINGS as binary
    from nodes.effects import NODE_DISPLAY_NAME_MAPPINGS as effects
    from nodes.export import NODE_DISPLAY_NAME_MAPPINGS as exp
    from nodes.generators import NODE_DISPLAY_NAME_MAPPINGS as gen
    from nodes.utility import NODE_DISPLAY_NAME_MAPPINGS as util
    from nodes.audio import NODE_DISPLAY_NAME_MAPPINGS as audio
    from nodes.flux_motion import NODE_DISPLAY_NAME_MAPPINGS as fm

    combined = {}
    for m in [dither, greyscale, binary, effects, exp, gen, util, audio, fm]:
        combined.update(m)
    return combined


class TestNodeRegistration:
    """Verify all expected node classes are registered."""

    def test_all_expected_registered(self):
        mapping = _gather_class_map()
        missing = EXPECTED - set(mapping.keys())
        assert not missing, f"Missing: {missing}"

    def test_every_key_has_display_name(self):
        mapping = _gather_class_map()
        names = _gather_name_map()
        missing = set(mapping.keys()) - set(names.keys())
        assert not missing, f"Without display name: {missing}"

    def test_minimum_count(self):
        mapping = _gather_class_map()
        assert len(mapping) >= 22, f"Only {len(mapping)} registered, expected >=22"


def _load_pack_like_comfyui():
    """Load the pack's top-level __init__ by file path, as ComfyUI does.

    The direct `from nodes.effects import ...` gathers above run with `nodes` as
    a real package. ComfyUI does not: it execs __init__.py, which then loads
    each category by file path with no parent package. Anything that only works
    under a real package import (e.g. `from ..utils import x`) silently drops
    out here, so this is the loader that has to be tested.
    """
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "koshi_pack_under_test",
        os.path.join(repo, "__init__.py"),
        submodule_search_locations=[repo],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestComfyUILoader:
    """The pack must expose every expected node through its real entry point."""

    def test_top_level_loader_registers_all_expected(self):
        pack = _load_pack_like_comfyui()
        missing = EXPECTED - set(pack.NODE_CLASS_MAPPINGS)
        assert not missing, f"Not registered via __init__.py: {sorted(missing)}"

    def test_top_level_loader_matches_direct_gather(self):
        pack = _load_pack_like_comfyui()
        assert set(pack.NODE_CLASS_MAPPINGS) == set(_gather_class_map())


class TestNodeInterface:
    """Verify every registered node has proper ComfyUI interface."""

    @pytest.fixture(scope="class")
    def node_map(self):
        return _gather_class_map()

    def test_has_function_attr(self, node_map):
        for key, cls in node_map.items():
            assert hasattr(cls, "FUNCTION"), f"{key} missing FUNCTION"
            fn = cls.FUNCTION
            has_method = hasattr(cls, fn) or callable(getattr(cls(), fn, None))
            assert has_method, f"{key}.{fn} not callable"

    def test_has_return_types(self, node_map):
        for key, cls in node_map.items():
            assert hasattr(cls, "RETURN_TYPES"), f"{key} missing RETURN_TYPES"
            rt = cls.RETURN_TYPES
            assert isinstance(rt, tuple), f"{key}.RETURN_TYPES must be tuple"

    def test_has_input_types(self, node_map):
        for key, cls in node_map.items():
            assert hasattr(cls, "INPUT_TYPES"), f"{key} missing INPUT_TYPES"
            assert callable(cls.INPUT_TYPES), f"{key}.INPUT_TYPES must be callable"

    def test_input_types_structure(self, node_map):
        for key, cls in node_map.items():
            result = cls.INPUT_TYPES()
            assert isinstance(result, dict), f"{key}.INPUT_TYPES() must return dict"
            assert "required" in result, f"{key} missing 'required' key"
            assert isinstance(result["required"], dict)

    def test_has_category(self, node_map):
        pfx = _PFX.replace("_", "")
        for key, cls in node_map.items():
            assert hasattr(cls, "CATEGORY"), f"{key} missing CATEGORY"
            assert isinstance(cls.CATEGORY, str)
            assert cls.CATEGORY.startswith(pfx), \
                f"{key}.CATEGORY='{cls.CATEGORY}' should start with '{pfx}'"
