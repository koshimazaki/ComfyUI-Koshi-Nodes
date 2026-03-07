"""Integration tests for Flux Motion pipeline: Schedule -> MotionEngine -> Feedback."""

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

from nodes.flux_motion.schedule import KoshiSchedule
from nodes.flux_motion.motion_engine import KoshiMotionEngine
from nodes.flux_motion.feedback import KoshiFeedback
from tests.mocks.comfyui import MockVAE


def _make_latent(batch=1, channels=4, height=8, width=8):
    torch.manual_seed(42)
    return {"samples": torch.randn(batch, channels, height, width, dtype=torch.float32)}


def _validate_latent(latent, expected_batch, channels=4, height=8, width=8):
    assert "samples" in latent
    samples = latent["samples"]
    assert samples.shape == (expected_batch, channels, height, width)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()


# ---------------------------------------------------------------------------
# 1. Schedule -> MotionEngine
# ---------------------------------------------------------------------------

class TestScheduleToEngine:

    def test_schedule_feeds_engine(self):
        """Parse a schedule and apply it to a latent via MotionEngine."""
        sched_node = KoshiSchedule()
        engine_node = KoshiMotionEngine()

        (schedule,) = sched_node.parse(
            schedule_string="0:(1.0), 9:(1.1)",
            max_frames=10,
            interpolation="linear",
            easing="none",
        )
        assert "values" in schedule
        assert len(schedule["values"]) == 10

        latent = _make_latent()
        (result,) = engine_node.process(
            latent=latent,
            zoom=schedule["values"][5],
            angle=0.0,
            translation_x=0.0,
            translation_y=0.0,
        )
        _validate_latent(result, 1)


# ---------------------------------------------------------------------------
# 2. MotionEngine feedback loop
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

        assert torch.isfinite(current["samples"]).all()
        assert not torch.allclose(latent["samples"], current["samples"])


# ---------------------------------------------------------------------------
# 3. MotionEngine output structure
# ---------------------------------------------------------------------------

class TestLatentDictStructure:

    def test_engine_output_has_samples_key(self):
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


# ---------------------------------------------------------------------------
# 4. Identity transform preserves latent
# ---------------------------------------------------------------------------

class TestStaticMotion:

    def test_identity_preserves_latent(self):
        engine_node = KoshiMotionEngine()
        latent = _make_latent()

        (result,) = engine_node.process(
            latent=latent,
            zoom=1.0,
            angle=0.0,
            translation_x=0.0,
            translation_y=0.0,
        )
        assert torch.allclose(
            latent["samples"], result["samples"], atol=1e-5
        )
