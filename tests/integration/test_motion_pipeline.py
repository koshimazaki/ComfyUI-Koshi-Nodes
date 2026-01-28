"""Integration tests for motion pipeline: schedule parsing, engine, and batch."""

import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch

from nodes.flux_motion.core.schedule_parser import MotionFrame
from nodes.flux_motion.schedule import KoshiScheduleMulti
from nodes.flux_motion.motion_engine import KoshiMotionEngine, KoshiMotionBatch
from nodes.flux_motion.semantic_motion import KoshiSemanticMotion
from tests.mocks.comfyui import MockVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latent(batch: int = 1, channels: int = 4, height: int = 8, width: int = 8):
    """Create a reproducible latent dict."""
    torch.manual_seed(42)
    return {"samples": torch.randn(batch, channels, height, width, dtype=torch.float32)}


def _validate_latent(latent: dict, expected_batch: int, channels: int = 4,
                     height: int = 8, width: int = 8):
    """Assert latent dict has valid structure and shape."""
    assert "samples" in latent, "Latent dict must contain 'samples' key"
    samples = latent["samples"]
    assert samples.shape == (expected_batch, channels, height, width), (
        f"Expected ({expected_batch}, {channels}, {height}, {width}), "
        f"got {tuple(samples.shape)}"
    )
    assert samples.dtype == torch.float32, f"Expected float32, got {samples.dtype}"
    assert torch.isfinite(samples).all(), "Latent contains non-finite values"


# ---------------------------------------------------------------------------
# 1. ScheduleMulti -> MotionEngine
# ---------------------------------------------------------------------------

class TestScheduleMultiToEngine:

    def test_schedule_feeds_engine(self):
        """Parse a multi-schedule and apply it to a latent via MotionEngine."""
        sched_node = KoshiScheduleMulti()
        engine_node = KoshiMotionEngine()

        # Stage 1: Parse schedule
        (schedule,) = sched_node.parse(
            max_frames=10,
            interpolation="linear",
            zoom="0:(1.0), 9:(1.1)",
            angle="0:(0), 9:(5)",
            translation_x="0:(0)",
            translation_y="0:(0)",
        )
        assert "motion_frames" in schedule
        assert len(schedule["motion_frames"]) == 10

        # Stage 2: Apply to latent at frame 5
        latent = _make_latent()
        (result,) = engine_node.process(
            latent=latent,
            zoom=1.0,
            angle=0.0,
            translation_x=0.0,
            translation_y=0.0,
            motion_schedule=schedule,
            frame_index=5,
        )
        _validate_latent(result, 1)


# ---------------------------------------------------------------------------
# 2. SemanticMotion -> MotionBatch
# ---------------------------------------------------------------------------

class TestSemanticMotionToBatch:

    def test_semantic_motion_feeds_batch(self):
        """Generate semantic motion schedule and apply via MotionBatch."""
        semantic_node = KoshiSemanticMotion()
        batch_node = KoshiMotionBatch()

        # Stage 1: Generate motion from text
        (schedule,) = semantic_node.generate(
            motion_prompt="slow zoom in",
            frames=4,
            intensity=1.0,
            easing="linear",
        )
        assert "motion_frames" in schedule
        assert len(schedule["motion_frames"]) == 4

        # Stage 2: Apply to batch latent
        latent = _make_latent(batch=4)
        (result,) = batch_node.process(
            latent=latent,
            motion_schedule=schedule,
            start_frame=0,
        )
        _validate_latent(result, 4)


# ---------------------------------------------------------------------------
# 3. SemanticMotion -> MotionEngine (single frame)
# ---------------------------------------------------------------------------

class TestSemanticMotionSingleFrame:

    def test_semantic_to_engine_single_frame(self):
        """Apply one frame of semantic motion through MotionEngine."""
        semantic_node = KoshiSemanticMotion()
        engine_node = KoshiMotionEngine()

        (schedule,) = semantic_node.generate(
            motion_prompt="pan right",
            frames=10,
            intensity=1.0,
            easing="linear",
        )

        latent = _make_latent()
        (result,) = engine_node.process(
            latent=latent,
            zoom=1.0,
            angle=0.0,
            translation_x=0.0,
            translation_y=0.0,
            motion_schedule=schedule,
            frame_index=0,
        )
        _validate_latent(result, 1)


# ---------------------------------------------------------------------------
# 4. Five-frame feedback loop
# ---------------------------------------------------------------------------

class TestFeedbackLoop:

    def test_five_frame_iterative_motion(self):
        """MotionEngine applied 5 times: output feeds back as input."""
        engine_node = KoshiMotionEngine()

        latent = _make_latent()
        current = latent

        for i in range(5):
            (current,) = engine_node.process(
                latent=current,
                zoom=1.02,
                angle=1.0,
                translation_x=1.0,
                translation_y=0.0,
            )
            _validate_latent(current, 1)

        # After 5 iterations the latent should still be finite
        assert torch.isfinite(current["samples"]).all()
        # It should differ from the original
        assert not torch.allclose(latent["samples"], current["samples"])


# ---------------------------------------------------------------------------
# 5. ScheduleMulti with zoom + rotation -> MotionBatch
# ---------------------------------------------------------------------------

class TestCombinedTransformsBatch:

    def test_zoom_and_rotation_batch(self):
        """Combined zoom and rotation schedule applied via MotionBatch."""
        sched_node = KoshiScheduleMulti()
        batch_node = KoshiMotionBatch()

        (schedule,) = sched_node.parse(
            max_frames=4,
            interpolation="linear",
            zoom="0:(1.0), 3:(1.15)",
            angle="0:(0), 3:(10)",
            translation_x="0:(0)",
            translation_y="0:(0)",
        )

        latent = _make_latent(batch=4)
        (result,) = batch_node.process(
            latent=latent,
            motion_schedule=schedule,
            start_frame=0,
        )
        _validate_latent(result, 4)

        # First frame should have zoom=1.0, angle=0 (identity transform)
        # so first sample should be unchanged from original
        assert torch.allclose(
            latent["samples"][0], result["samples"][0], atol=1e-5
        ), "Frame 0 with identity transform should be approximately unchanged"


# ---------------------------------------------------------------------------
# 6. Each chain produces valid latent dict
# ---------------------------------------------------------------------------

class TestLatentDictStructure:

    def test_engine_output_has_samples_key(self):
        """MotionEngine output must be a dict with 'samples' tensor."""
        engine_node = KoshiMotionEngine()
        latent = _make_latent()

        (result,) = engine_node.process(
            latent=latent,
            zoom=1.05,
            angle=2.0,
            translation_x=3.0,
            translation_y=-1.0,
        )
        assert isinstance(result, dict)
        assert "samples" in result
        assert isinstance(result["samples"], torch.Tensor)

    def test_batch_output_has_samples_key(self):
        """MotionBatch output must be a dict with 'samples' tensor."""
        semantic_node = KoshiSemanticMotion()
        batch_node = KoshiMotionBatch()

        (schedule,) = semantic_node.generate(
            motion_prompt="rotate left",
            frames=2,
        )
        latent = _make_latent(batch=2)
        (result,) = batch_node.process(
            latent=latent,
            motion_schedule=schedule,
        )
        assert isinstance(result, dict)
        assert "samples" in result


# ---------------------------------------------------------------------------
# 7. Batch sizes propagate correctly
# ---------------------------------------------------------------------------

class TestBatchSizePropagation:

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_preserved(self, batch_size):
        """MotionBatch preserves input batch size in output."""
        semantic_node = KoshiSemanticMotion()
        batch_node = KoshiMotionBatch()

        (schedule,) = semantic_node.generate(
            motion_prompt="slow zoom in",
            frames=batch_size,
        )
        latent = _make_latent(batch=batch_size)
        (result,) = batch_node.process(
            latent=latent,
            motion_schedule=schedule,
        )
        assert result["samples"].shape[0] == batch_size


# ---------------------------------------------------------------------------
# 8. Static motion produces approximately unchanged latent
# ---------------------------------------------------------------------------

class TestStaticMotion:

    def test_static_motion_identity(self):
        """A 'static' semantic motion should not change the latent."""
        semantic_node = KoshiSemanticMotion()
        engine_node = KoshiMotionEngine()

        (schedule,) = semantic_node.generate(
            motion_prompt="static",
            frames=5,
        )

        latent = _make_latent()
        (result,) = engine_node.process(
            latent=latent,
            zoom=1.0,
            angle=0.0,
            translation_x=0.0,
            translation_y=0.0,
            motion_schedule=schedule,
            frame_index=0,
        )

        # Static = zoom 1.0, angle 0, translation 0 -> identity
        assert torch.allclose(
            latent["samples"], result["samples"], atol=1e-5
        ), "Static motion should produce unchanged latent"

    def test_static_batch_all_frames_unchanged(self):
        """Static motion through MotionBatch: all frames stay unchanged."""
        semantic_node = KoshiSemanticMotion()
        batch_node = KoshiMotionBatch()

        (schedule,) = semantic_node.generate(
            motion_prompt="still",
            frames=4,
        )

        latent = _make_latent(batch=4)
        (result,) = batch_node.process(
            latent=latent,
            motion_schedule=schedule,
        )

        assert torch.allclose(
            latent["samples"], result["samples"], atol=1e-5
        ), "Static batch motion should produce unchanged latents"
