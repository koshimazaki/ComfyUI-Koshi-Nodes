"""Tests for nodes.flux_motion.v2v_nodes -- V2V utility nodes."""

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
import warnings

from nodes.flux_motion.v2v_nodes import (
    KoshiColorMatchLAB,
    KoshiOpticalFlowWarp,
    KoshiImageBlend,
    KoshiV2VProcessor,
    KoshiV2VMetadata,
)
from tests.mocks.comfyui import MockVAE, MockModel, mock_conditioning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def color_match_node():
    return KoshiColorMatchLAB()


@pytest.fixture
def flow_warp_node():
    return KoshiOpticalFlowWarp()


@pytest.fixture
def blend_node():
    return KoshiImageBlend()


@pytest.fixture
def v2v_proc_node():
    return KoshiV2VProcessor()


@pytest.fixture
def v2v_meta_node():
    return KoshiV2VMetadata()


@pytest.fixture
def image_a():
    """Random image [1, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def image_b():
    """Different random image [1, 64, 64, 3]."""
    torch.manual_seed(99)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def batch_images():
    """Batch of 4 images [4, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def anchor_image():
    """Anchor image for colour matching [1, 64, 64, 3]."""
    torch.manual_seed(77)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


# ===================================================================
# KoshiColorMatchLAB
# ===================================================================

class TestColorMatchOutputShape:
    """Output shape matches input."""

    def test_shape_preserved(self, color_match_node, image_a, anchor_image):
        result = color_match_node.match(
            image=image_a, anchor=anchor_image, strength=0.5,
        )
        assert result[0].shape == image_a.shape


class TestColorMatchStrengthZero:
    """strength=0 returns input unchanged."""

    def test_strength_zero_identity(self, color_match_node, image_a, anchor_image):
        result = color_match_node.match(
            image=image_a, anchor=anchor_image, strength=0.0,
        )
        assert torch.allclose(result[0], image_a, atol=1e-5)


class TestColorMatchOutputRange:
    """Output values in [0, 1]."""

    def test_output_range(self, color_match_node, image_a, anchor_image):
        result = color_match_node.match(
            image=image_a, anchor=anchor_image, strength=1.0,
        )
        assert result[0].min().item() >= -0.01
        assert result[0].max().item() <= 1.01


class TestColorMatchBatch:
    """Batch processing works correctly."""

    def test_batch_processing(self, color_match_node, batch_images, anchor_image):
        result = color_match_node.match(
            image=batch_images, anchor=anchor_image, strength=0.8,
        )
        assert result[0].shape == batch_images.shape


# ===================================================================
# KoshiOpticalFlowWarp
# ===================================================================

class TestOpticalFlowOutputShape:
    """Output shape matches input (requires cv2)."""

    @pytest.mark.cv2
    def test_shape_preserved(self, flow_warp_node, image_a, image_b):
        cv2 = pytest.importorskip("cv2")
        result = flow_warp_node.warp(
            image_to_warp=image_a, flow_from=image_a, flow_to=image_b,
        )
        assert result[0].shape == image_a.shape


class TestOpticalFlowBatch:
    """Batch processing (requires cv2)."""

    @pytest.mark.cv2
    def test_batch_warp(self, flow_warp_node, batch_images):
        cv2 = pytest.importorskip("cv2")
        # Use same batch as flow_from and shifted as flow_to
        flow_to = torch.roll(batch_images, shifts=1, dims=0)
        result = flow_warp_node.warp(
            image_to_warp=batch_images,
            flow_from=batch_images,
            flow_to=flow_to,
        )
        assert result[0].shape == batch_images.shape


# ===================================================================
# KoshiImageBlend
# ===================================================================

class TestBlendAlphaZero:
    """alpha=0 returns image1."""

    def test_alpha_zero_returns_image1(self, blend_node, image_a, image_b):
        result = blend_node.blend(image1=image_a, image2=image_b, alpha=0.0)
        assert torch.allclose(result[0], image_a, atol=1e-5)


class TestBlendAlphaOne:
    """alpha=1 returns image2."""

    def test_alpha_one_returns_image2(self, blend_node, image_a, image_b):
        result = blend_node.blend(image1=image_a, image2=image_b, alpha=1.0)
        assert torch.allclose(result[0], image_b, atol=1e-5)


class TestBlendAlphaHalf:
    """alpha=0.5 returns average of inputs."""

    def test_alpha_half_average(self, blend_node, image_a, image_b):
        result = blend_node.blend(image1=image_a, image2=image_b, alpha=0.5)
        expected = image_a * 0.5 + image_b * 0.5
        assert torch.allclose(result[0], expected, atol=1e-5)


class TestBlendOutputShape:
    """Output shape matches input."""

    def test_output_shape(self, blend_node, image_a, image_b):
        result = blend_node.blend(image1=image_a, image2=image_b, alpha=0.5)
        assert result[0].shape == image_a.shape


class TestBlendMask:
    """Mask-based blending: zero mask returns image1, ones mask returns image2."""

    def test_mask_zeros_returns_image1(self, blend_node, image_a, image_b):
        mask = torch.zeros(1, 64, 64, dtype=torch.float32)
        result = blend_node.blend(
            image1=image_a, image2=image_b, alpha=0.5, mask=mask,
        )
        assert torch.allclose(result[0], image_a, atol=1e-5)

    def test_mask_ones_returns_image2(self, blend_node, image_a, image_b):
        mask = torch.ones(1, 64, 64, dtype=torch.float32)
        result = blend_node.blend(
            image1=image_a, image2=image_b, alpha=0.5, mask=mask,
        )
        assert torch.allclose(result[0], image_b, atol=1e-5)


# ===================================================================
# KoshiV2VProcessor (DEPRECATED)
# ===================================================================

class TestV2VProcessorDeprecated:
    """Deprecated node emits warning and returns input unchanged."""

    def test_emits_deprecation_warning(self, v2v_proc_node, image_a):
        model = MockModel()
        vae = MockVAE()
        cond = mock_conditioning()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v2v_proc_node.process(
                images=image_a, model=model, positive=cond,
                negative=cond, vae=vae, mode="pure",
                denoise=0.65, steps=20, cfg=3.5, seed=42,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

    def test_returns_input_unchanged(self, v2v_proc_node, image_a):
        model = MockModel()
        vae = MockVAE()
        cond = mock_conditioning()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = v2v_proc_node.process(
                images=image_a, model=model, positive=cond,
                negative=cond, vae=vae, mode="pure",
                denoise=0.65, steps=20, cfg=3.5, seed=42,
            )
        assert torch.allclose(result[0], image_a, atol=1e-5)


# ===================================================================
# KoshiV2VMetadata
# ===================================================================

class TestV2VMetadata:
    """V2V metadata node saves JSON file."""

    def test_saves_metadata_file(self, v2v_meta_node, tmp_path):
        output_path = str(tmp_path)
        result = v2v_meta_node.save(
            preset="test_preset",
            output_path=output_path,
        )
        assert "ui" in result
        # Verify a JSON file was created in the output path
        import glob
        json_files = glob.glob(os.path.join(output_path, "*.json"))
        assert len(json_files) >= 1
