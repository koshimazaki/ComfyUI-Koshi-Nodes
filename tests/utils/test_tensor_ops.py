"""Tests for nodes.utils.tensor_ops -- ComfyUI format conversion utilities."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nodes.utils.tensor_ops import to_comfy_image, from_comfy_image, ensure_4d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chw(c=3, h=16, w=16):
    """Create a (C, H, W) tensor with known values."""
    torch.manual_seed(0)
    return torch.rand(c, h, w, dtype=torch.float32)


def _make_bchw(b=2, c=3, h=16, w=16):
    """Create a (B, C, H, W) tensor with known values."""
    torch.manual_seed(0)
    return torch.rand(b, c, h, w, dtype=torch.float32)


def _make_bhwc(b=2, h=16, w=16, c=3):
    """Create a (B, H, W, C) tensor in ComfyUI layout."""
    torch.manual_seed(0)
    return torch.rand(b, h, w, c, dtype=torch.float32)


# ---------------------------------------------------------------------------
# to_comfy_image
# ---------------------------------------------------------------------------

class TestToComfyImage:
    """Verify conversion *to* ComfyUI (B, H, W, C) format."""

    def test_3d_chw_to_bhwc(self):
        """3D (C,H,W) input should become (1,H,W,C)."""
        src = _make_chw(c=3, h=8, w=12)
        result = to_comfy_image(src)
        assert result.shape == (1, 8, 12, 3)

    def test_4d_bchw_to_bhwc(self):
        """4D (B,C,H,W) input should become (B,H,W,C)."""
        src = _make_bchw(b=4, c=3, h=8, w=12)
        result = to_comfy_image(src)
        assert result.shape == (4, 8, 12, 3)

    def test_single_channel_3d(self):
        """Single-channel (1,H,W) should produce (1,H,W,1)."""
        src = _make_chw(c=1, h=10, w=10)
        result = to_comfy_image(src)
        assert result.shape == (1, 10, 10, 1)

    def test_output_clamped_to_unit_range(self):
        """Values outside [0, 1] must be clamped after conversion."""
        src = torch.tensor([[[-0.5, 2.0], [0.3, 0.7]]])  # (1, 2, 2)
        result = to_comfy_image(src)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_preserved_float32(self):
        """Output dtype should remain float32."""
        src = _make_bchw(b=1, c=3, h=4, w=4)
        result = to_comfy_image(src)
        assert result.dtype == torch.float32

    def test_rgba_4_channels(self):
        """4-channel (B,4,H,W) should still permute correctly."""
        src = _make_bchw(b=1, c=4, h=6, w=6)
        result = to_comfy_image(src)
        assert result.shape == (1, 6, 6, 4)


# ---------------------------------------------------------------------------
# from_comfy_image
# ---------------------------------------------------------------------------

class TestFromComfyImage:
    """Verify conversion *from* ComfyUI (B, H, W, C) back to (B, C, H, W)."""

    def test_4d_bhwc_to_bchw(self):
        """Standard ComfyUI (B,H,W,C) with C=3 should become (B,C,H,W)."""
        src = _make_bhwc(b=2, h=8, w=12, c=3)
        result = from_comfy_image(src)
        assert result.shape == (2, 3, 8, 12)

    def test_single_channel_bhwc(self):
        """(B,H,W,1) should become (B,1,H,W)."""
        src = _make_bhwc(b=1, h=10, w=10, c=1)
        result = from_comfy_image(src)
        assert result.shape == (1, 1, 10, 10)

    def test_passthrough_when_last_dim_large(self):
        """If last dim > 4 the tensor is returned unchanged (not channel-last)."""
        src = torch.rand(2, 8, 8, 8, dtype=torch.float32)  # last dim 8
        result = from_comfy_image(src)
        assert result.shape == src.shape, "Should pass through when last dim > 4"


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Converting to ComfyUI format and back should preserve data."""

    def test_to_then_from_preserves_data(self):
        """to_comfy_image -> from_comfy_image should recover the original layout."""
        original = torch.rand(2, 3, 16, 16, dtype=torch.float32)
        comfy = to_comfy_image(original.clone())
        recovered = from_comfy_image(comfy)
        assert recovered.shape == original.shape
        # Clamping may change out-of-range values, but rand is [0,1] so data matches
        assert torch.allclose(recovered, original, atol=1e-6)

    def test_from_then_to_preserves_data(self):
        """from_comfy_image -> to_comfy_image should recover the ComfyUI layout."""
        original = torch.rand(2, 16, 16, 3, dtype=torch.float32)
        standard = from_comfy_image(original.clone())
        recovered = to_comfy_image(standard)
        assert recovered.shape == original.shape
        assert torch.allclose(recovered, original, atol=1e-6)


# ---------------------------------------------------------------------------
# ensure_4d
# ---------------------------------------------------------------------------

class TestEnsure4D:
    """Verify dimension expansion to 4D."""

    def test_2d_to_4d(self):
        """(H, W) should become (1, 1, H, W)."""
        src = torch.rand(8, 8)
        result = ensure_4d(src)
        assert result.dim() == 4
        assert result.shape == (1, 1, 8, 8)

    def test_3d_to_4d(self):
        """(C, H, W) should become (1, C, H, W)."""
        src = torch.rand(3, 8, 8)
        result = ensure_4d(src)
        assert result.dim() == 4
        assert result.shape == (1, 3, 8, 8)

    def test_4d_unchanged(self):
        """Already-4D tensor should pass through without modification."""
        src = torch.rand(2, 3, 8, 8)
        result = ensure_4d(src)
        assert result.shape == src.shape
        assert torch.equal(result, src)


# ---------------------------------------------------------------------------
# Batch size variations
# ---------------------------------------------------------------------------

class TestBatchSizes:
    """Verify that various batch sizes work correctly through conversion."""

    @pytest.mark.parametrize("batch", [1, 2, 5, 16])
    def test_to_comfy_various_batches(self, batch):
        src = torch.rand(batch, 3, 8, 8, dtype=torch.float32)
        result = to_comfy_image(src)
        assert result.shape == (batch, 8, 8, 3)

    @pytest.mark.parametrize("batch", [1, 2, 5, 16])
    def test_from_comfy_various_batches(self, batch):
        src = torch.rand(batch, 8, 8, 3, dtype=torch.float32)
        result = from_comfy_image(src)
        assert result.shape == (batch, 3, 8, 8)
