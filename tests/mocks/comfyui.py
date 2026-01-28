"""Mock ComfyUI objects for testing nodes without ComfyUI runtime."""

import torch
import numpy as np


class MockVAE:
    """Mock VAE that simulates encode/decode without actual model."""

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate VAE encoding: (B, H, W, C) -> latent (B, 4, H//8, W//8)."""
        if image.dim() == 4:
            b, h, w, c = image.shape
            return torch.randn(b, 4, h // 8, w // 8, device=image.device)
        elif image.dim() == 3:
            h, w, c = image.shape
            return torch.randn(1, 4, h // 8, w // 8, device=image.device)
        return torch.randn(1, 4, 8, 8)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Simulate VAE decoding: (B, 4, H, W) -> image (B, H*8, W*8, 3)."""
        b, c, h, w = latent.shape
        return torch.rand(b, h * 8, w * 8, 3, device=latent.device)


class MockModel:
    """Mock diffusion model for testing nodes that require MODEL input."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model_options = {}

    def get_model_object(self, name):
        return None

    def model_dtype(self):
        return torch.float32


class MockCLIP:
    """Mock CLIP model for testing nodes that require CLIP input."""

    def encode_from_tokens(self, tokens, return_pooled=False):
        cond = torch.randn(1, 77, 768)
        pooled = {"pooled_output": torch.randn(1, 768)}
        return cond, pooled

    def tokenize(self, text):
        return {"l": [[49406] + [0] * 76]}


def mock_conditioning():
    """Create mock conditioning tuple as used by ComfyUI."""
    cond = torch.randn(1, 77, 768)
    pooled = {"pooled_output": torch.randn(1, 768)}
    return [[cond, pooled]]
