"""Shared fixtures for ComfyUI-Koshi-Nodes test suite."""

import sys
import os
import pytest
import torch
import numpy as np

# Ensure the package root is importable
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Also add parent for Deforum2026 cross-verify imports
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.mocks.comfyui import MockVAE, MockModel, MockCLIP, mock_conditioning


# ---------------------------------------------------------------------------
# Image fixtures (ComfyUI format: B, H, W, C float32 [0,1])
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_rgb():
    """Single 64x64 RGB image in ComfyUI format [1, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def sample_image_batch():
    """Batch of 4 RGB images [4, 64, 64, 3]."""
    torch.manual_seed(42)
    return torch.rand(4, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def gradient_image():
    """Horizontal gradient image [1, 64, 64, 3] — left=black, right=white."""
    grad = torch.linspace(0, 1, 64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    grad = grad.expand(1, 64, 64, 3)
    return grad.clone()


@pytest.fixture
def black_image():
    """All-zeros image [1, 64, 64, 3]."""
    return torch.zeros(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def white_image():
    """All-ones image [1, 64, 64, 3]."""
    return torch.ones(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def sample_latent():
    """Sample latent dict as used by ComfyUI {'samples': Tensor(1,4,8,8)}."""
    torch.manual_seed(42)
    return {"samples": torch.randn(1, 4, 8, 8, dtype=torch.float32)}


@pytest.fixture
def sample_latent_batch():
    """Batch latent dict {'samples': Tensor(4,4,8,8)}."""
    torch.manual_seed(42)
    return {"samples": torch.randn(4, 4, 8, 8, dtype=torch.float32)}


@pytest.fixture
def sample_mask():
    """Single mask [1, 64, 64] float32 [0,1]."""
    torch.manual_seed(42)
    return torch.rand(1, 64, 64, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Mock ComfyUI objects
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vae():
    """Mock VAE for encode/decode."""
    return MockVAE()


@pytest.fixture
def mock_model():
    """Mock diffusion model."""
    return MockModel()


@pytest.fixture
def mock_clip():
    """Mock CLIP model."""
    return MockCLIP()


@pytest.fixture
def mock_cond():
    """Mock conditioning."""
    return mock_conditioning()
