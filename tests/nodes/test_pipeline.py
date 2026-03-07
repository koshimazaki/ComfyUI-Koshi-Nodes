"""Tests for nodes.flux_motion.pipeline -- KoshiFrameIterator."""

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

from nodes.flux_motion.pipeline import KoshiFrameIterator
from nodes.flux_motion.core.schedule_parser import MotionFrame
from tests.mocks.comfyui import MockVAE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iterator_node():
    return KoshiFrameIterator()


@pytest.fixture
def vae():
    return MockVAE()


@pytest.fixture
def batch_images():
    """Batch of 4 images [4, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def motion_schedule_4():
    """Motion schedule with 4 frames."""
    return {
        "frames": 4,
        "motion_frames": [
            MotionFrame(frame_index=i, zoom=1.0 + 0.02 * i, strength=0.5 + 0.1 * i)
            for i in range(4)
        ],
        "interpolation": "linear",
    }


# ===================================================================
# KoshiFrameIterator
# ===================================================================

class TestIteratorCorrectFrame:
    """Returns correct frame from batch at given index."""

    def test_returns_frame_at_index(self, iterator_node, batch_images, motion_schedule_4, vae):
        result = iterator_node.iterate(
            images=batch_images, motion_schedule=motion_schedule_4,
            frame_index=2, vae=vae,
        )
        image_out = result[0]
        expected = batch_images[2:3]
        assert torch.allclose(image_out, expected, atol=1e-5)


class TestIteratorFirstFrame:
    """frame_index=0 returns first frame."""

    def test_first_frame(self, iterator_node, batch_images, motion_schedule_4, vae):
        result = iterator_node.iterate(
            images=batch_images, motion_schedule=motion_schedule_4,
            frame_index=0, vae=vae,
        )
        image_out = result[0]
        expected = batch_images[0:1]
        assert torch.allclose(image_out, expected, atol=1e-5)


class TestIteratorRemainingSchedule:
    """Returns the full motion schedule as remaining_schedule."""

    def test_remaining_schedule(self, iterator_node, batch_images, motion_schedule_4, vae):
        result = iterator_node.iterate(
            images=batch_images, motion_schedule=motion_schedule_4,
            frame_index=1, vae=vae,
        )
        remaining = result[4]  # 5th output: remaining_schedule
        assert remaining is motion_schedule_4


class TestIteratorFrameIndex:
    """Returns correct frame_index integer."""

    def test_frame_index_output(self, iterator_node, batch_images, motion_schedule_4, vae):
        for idx in range(4):
            result = iterator_node.iterate(
                images=batch_images, motion_schedule=motion_schedule_4,
                frame_index=idx, vae=vae,
            )
            assert result[2] == idx


class TestIteratorStrength:
    """Returns strength value from the motion schedule."""

    def test_strength_from_schedule(self, iterator_node, batch_images, motion_schedule_4, vae):
        result = iterator_node.iterate(
            images=batch_images, motion_schedule=motion_schedule_4,
            frame_index=0, vae=vae,
        )
        strength = result[3]
        # First frame: strength = 0.5 + 0.1 * 0 = 0.5
        assert abs(strength - 0.5) < 1e-4


class TestIteratorBatchOf4:
    """Works with batch of 4 frames, extracting each sequentially."""

    def test_batch_of_4(self, iterator_node, batch_images, motion_schedule_4, vae):
        for idx in range(4):
            result = iterator_node.iterate(
                images=batch_images, motion_schedule=motion_schedule_4,
                frame_index=idx, vae=vae,
            )
            # Returns 5-tuple: image, latent_dict, frame_index, strength, schedule
            assert len(result) == 5
            assert result[0].shape == (1, 64, 64, 3)
            assert isinstance(result[1], dict)
            assert "samples" in result[1]
            assert result[2] == idx
            assert isinstance(result[3], float)
