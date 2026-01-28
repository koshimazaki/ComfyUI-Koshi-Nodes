"""Sprint 2: Verify all modules import correctly."""

import sys
import os
import pytest

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreImports:
    """Test that core algorithm modules import with only torch+numpy."""

    def test_import_easing(self):
        from nodes.flux_motion.core.easing import (
            apply_easing, apply_easing_to_range, EASING_PRESETS, list_easings
        )
        assert callable(apply_easing)
        assert isinstance(EASING_PRESETS, dict)

    def test_import_interpolation(self):
        from nodes.flux_motion.core.interpolation import (
            linear_interpolation, step_interpolation, interpolate_array, lerp, smoothstep
        )
        assert callable(linear_interpolation)
        assert callable(lerp)

    def test_import_schedule_parser(self):
        from nodes.flux_motion.core.schedule_parser import (
            MotionFrame, parse_schedule_string, parse_deforum_params
        )
        assert callable(parse_schedule_string)

    def test_import_transforms(self):
        from nodes.flux_motion.core.transforms import (
            create_affine_matrix, apply_affine_transform, apply_composite_transform
        )
        assert callable(create_affine_matrix)

    def test_import_core_package(self):
        from nodes.flux_motion.core import (
            linear_interpolation, apply_easing, create_affine_matrix,
            MotionFrame, EASING_PRESETS
        )
        assert callable(linear_interpolation)


class TestUtilImports:
    """Test utility module imports."""

    def test_import_tensor_ops(self):
        from nodes.utils.tensor_ops import to_comfy_image, from_comfy_image, ensure_4d
        assert callable(to_comfy_image)

    def test_import_metadata(self):
        from nodes.utils.metadata import (
            capture_settings, save_metadata, load_metadata, metadata_to_string
        )
        assert callable(capture_settings)


class TestCategoryImports:
    """Test that each node category module imports without error."""

    def test_import_dither(self):
        from nodes.image.dither import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_greyscale(self):
        from nodes.image.greyscale import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_binary(self):
        from nodes.image.binary import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_effects(self):
        from nodes.effects import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_export(self):
        from nodes.export import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_generators(self):
        from nodes.generators import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_utility(self):
        from nodes.utility import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_flux_motion(self):
        from nodes.flux_motion import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)

    def test_import_audio(self):
        from nodes.audio import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)
        assert len(NODE_CLASS_MAPPINGS) == 0  # Empty, but importable


class TestGracefulDegradation:
    """Test that optional dependency flags exist and modules handle missing deps."""

    def test_bloom_moderngl_flag(self):
        from nodes.effects.bloom import MODERNGL_AVAILABLE
        assert isinstance(MODERNGL_AVAILABLE, bool)

    def test_bloom_scipy_flag(self):
        from nodes.effects.bloom import SCIPY_AVAILABLE
        assert isinstance(SCIPY_AVAILABLE, bool)

    def test_chromatic_aberration_flags(self):
        from nodes.effects.chromatic_aberration import MODERNGL_AVAILABLE, SCIPY_AVAILABLE
        assert isinstance(MODERNGL_AVAILABLE, bool)
        assert isinstance(SCIPY_AVAILABLE, bool)

    def test_hologram_scipy_flag(self):
        from nodes.effects.hologram import SCIPY_AVAILABLE
        assert isinstance(SCIPY_AVAILABLE, bool)

    def test_oled_screen_pil_flag(self):
        from nodes.export.oled_screen import PIL_AVAILABLE
        assert isinstance(PIL_AVAILABLE, bool)
