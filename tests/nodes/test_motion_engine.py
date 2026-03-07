"""Tests for nodes.flux_motion.motion_engine -- KoshiMotionEngine."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch

from nodes.flux_motion.motion_engine import KoshiMotionEngine


@pytest.fixture
def engine():
    """KoshiMotionEngine instance."""
    return KoshiMotionEngine()


@pytest.fixture
def latent():
    """Single latent dict {samples: (1,4,8,8)}."""
    torch.manual_seed(42)
    return {"samples": torch.randn(1, 4, 8, 8, dtype=torch.float32)}


# ===================================================================
# KoshiMotionEngine
# ===================================================================

class TestMotionEngineIdentity:
    """Identity transform (zoom=1, angle=0, tx=0, ty=0) preserves input."""

    def test_identity_preserves_input(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.0, angle=0.0,
            translation_x=0.0, translation_y=0.0,
        )
        output = result[0]["samples"]
        assert torch.allclose(output, latent["samples"], atol=1e-5)


class TestMotionEngineZoom:
    """Non-unity zoom modifies the output."""

    def test_zoom_changes_output(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.5, angle=0.0,
            translation_x=0.0, translation_y=0.0,
        )
        output = result[0]["samples"]
        assert not torch.allclose(output, latent["samples"], atol=1e-3)


class TestMotionEngineRotation:
    """Non-zero angle modifies the output."""

    def test_rotation_changes_output(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.0, angle=45.0,
            translation_x=0.0, translation_y=0.0,
        )
        output = result[0]["samples"]
        assert not torch.allclose(output, latent["samples"], atol=1e-3)


class TestMotionEngineTranslation:
    """Non-zero translation modifies the output."""

    def test_translation_changes_output(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.0, angle=0.0,
            translation_x=10.0, translation_y=0.0,
        )
        output = result[0]["samples"]
        assert not torch.allclose(output, latent["samples"], atol=1e-3)


class TestMotionEngineOutputShape:
    """Output shape matches input shape."""

    def test_output_shape_matches_input(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.2, angle=15.0,
            translation_x=5.0, translation_y=3.0,
        )
        output = result[0]["samples"]
        assert output.shape == latent["samples"].shape


class TestMotionEngineOutputDict:
    """Output is a dict with 'samples' key."""

    def test_output_is_latent_dict(self, engine, latent):
        result = engine.process(
            latent=latent, zoom=1.0, angle=0.0,
            translation_x=0.0, translation_y=0.0,
        )
        assert isinstance(result[0], dict)
        assert "samples" in result[0]


class TestMotionEngineMask:
    """Motion mask blends selectively between original and transformed."""

    def test_mask_blending(self, engine, latent):
        # Mask of zeros -> output should equal original (no transform applied)
        mask = torch.zeros(1, 8, 8, dtype=torch.float32)
        result = engine.process(
            latent=latent, zoom=1.5, angle=30.0,
            translation_x=10.0, translation_y=10.0,
            motion_mask=mask,
        )
        output = result[0]["samples"]
        assert torch.allclose(output, latent["samples"], atol=1e-5)


