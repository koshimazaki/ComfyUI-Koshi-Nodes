"""Tests for nodes.flux_motion.motion_engine -- KoshiMotionEngine and KoshiMotionBatch."""

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

from nodes.flux_motion.motion_engine import KoshiMotionEngine, KoshiMotionBatch
from nodes.flux_motion.core.schedule_parser import MotionFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """KoshiMotionEngine instance."""
    return KoshiMotionEngine()


@pytest.fixture
def batch_engine():
    """KoshiMotionBatch instance."""
    return KoshiMotionBatch()


@pytest.fixture
def latent():
    """Single latent dict {samples: (1,4,8,8)}."""
    torch.manual_seed(42)
    return {"samples": torch.randn(1, 4, 8, 8, dtype=torch.float32)}


@pytest.fixture
def latent_batch():
    """Batch latent dict {samples: (4,4,8,8)}."""
    torch.manual_seed(42)
    return {"samples": torch.randn(4, 4, 8, 8, dtype=torch.float32)}


@pytest.fixture
def motion_schedule_4():
    """Motion schedule with 4 frames, slight zoom."""
    return {
        "motion_frames": [
            MotionFrame(frame_index=i, zoom=1.0 + 0.05 * i)
            for i in range(4)
        ]
    }


@pytest.fixture
def identity_schedule_4():
    """Motion schedule with 4 frames, all identity transforms."""
    return {
        "motion_frames": [
            MotionFrame(frame_index=i, zoom=1.0, angle=0.0,
                        translation_x=0.0, translation_y=0.0)
            for i in range(4)
        ]
    }


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


# ===================================================================
# KoshiMotionBatch
# ===================================================================

class TestMotionBatchOutputBatchSize:
    """Output batch equals input batch size."""

    def test_output_batch_size(self, batch_engine, latent_batch, motion_schedule_4):
        result = batch_engine.process(
            latent=latent_batch,
            motion_schedule=motion_schedule_4,
        )
        output = result[0]["samples"]
        assert output.shape[0] == latent_batch["samples"].shape[0]


class TestMotionBatchStartFrameOffset:
    """Start frame offsets into the schedule correctly."""

    def test_start_frame_offset(self, batch_engine, latent_batch, motion_schedule_4):
        result_0 = batch_engine.process(
            latent=latent_batch, motion_schedule=motion_schedule_4, start_frame=0,
        )
        result_2 = batch_engine.process(
            latent=latent_batch, motion_schedule=motion_schedule_4, start_frame=2,
        )
        # Different start frames should produce different outputs
        # (start_frame=2 exceeds schedule length for later frames, using identity)
        out_0 = result_0[0]["samples"]
        out_2 = result_2[0]["samples"]
        assert not torch.allclose(out_0, out_2, atol=1e-3)


class TestMotionBatchVaryingZoom:
    """Schedule with varying zoom produces different frames."""

    def test_varying_zoom_different_frames(self, batch_engine, latent_batch, motion_schedule_4):
        result = batch_engine.process(
            latent=latent_batch,
            motion_schedule=motion_schedule_4,
        )
        output = result[0]["samples"]
        # First frame (zoom=1.0) should differ from last frame (zoom=1.15)
        assert not torch.allclose(output[0], output[3], atol=1e-3)


class TestMotionBatchSingleFrame:
    """Single-frame schedule works correctly."""

    def test_single_frame_schedule(self, batch_engine):
        latent = {"samples": torch.randn(1, 4, 8, 8)}
        schedule = {
            "motion_frames": [MotionFrame(frame_index=0, zoom=1.1)]
        }
        result = batch_engine.process(latent=latent, motion_schedule=schedule)
        output = result[0]
        assert isinstance(output, dict)
        assert "samples" in output
        assert output["samples"].shape[0] == 1


class TestMotionBatchOutputDict:
    """Output is dict with 'samples' key."""

    def test_output_is_latent_dict(self, batch_engine, latent_batch, motion_schedule_4):
        result = batch_engine.process(
            latent=latent_batch,
            motion_schedule=motion_schedule_4,
        )
        assert isinstance(result[0], dict)
        assert "samples" in result[0]
