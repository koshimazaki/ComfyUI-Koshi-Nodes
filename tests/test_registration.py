"""Sprint 2: Verify all node classes register correctly with proper ComfyUI interface."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build expected node keys programmatically
_PFX = "Koshi_"
_SUFFIXES = [
    # Image processing
    "Dither", "Greyscale", "Binary",
    # Effects
    "Bloom", "ChromaticAberration", "Hologram", "Scanlines", "VideoGlitch", "Glitch",
    # Export
    "SIDKITScreen", "OLEDPreview", "PixelScaler", "SpriteSheet", "OLEDScreen", "XBMExport",
    # Generators
    "GlitchCandies", "ShapeMorph", "NoiseDisplace",
    # Utility
    "CaptureSettings", "SaveMetadata", "DisplayMetadata",
    # Flux Motion
    "Schedule", "ScheduleMulti", "MotionEngine", "MotionBatch",
    "SemanticMotion", "Feedback", "FeedbackSimple",
    "AnimationPipeline", "FrameIterator",
    # V2V
    "ColorMatchLAB", "OpticalFlowWarp", "ImageBlend", "V2VProcessor", "V2VMetadata",
]
EXPECTED = {_PFX + s for s in _SUFFIXES}


def _gather_class_map():
    """Collect NODE_CLASS_MAPPINGS from all categories."""
    from nodes.image.dither import NODE_CLASS_MAPPINGS as dither
    from nodes.image.greyscale import NODE_CLASS_MAPPINGS as greyscale
    from nodes.image.binary import NODE_CLASS_MAPPINGS as binary
    from nodes.effects import NODE_CLASS_MAPPINGS as effects
    from nodes.export import NODE_CLASS_MAPPINGS as exp
    from nodes.generators import NODE_CLASS_MAPPINGS as gen
    from nodes.utility import NODE_CLASS_MAPPINGS as util
    from nodes.flux_motion import NODE_CLASS_MAPPINGS as fm

    combined = {}
    for m in [dither, greyscale, binary, effects, exp, gen, util, fm]:
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
    from nodes.flux_motion import NODE_DISPLAY_NAME_MAPPINGS as fm

    combined = {}
    for m in [dither, greyscale, binary, effects, exp, gen, util, fm]:
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
        assert len(mapping) >= 30, f"Only {len(mapping)} registered, expected >=30"


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
