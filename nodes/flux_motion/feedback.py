"""Koshi Feedback Processor - Frame-to-frame coherence enhancement."""

import torch
import numpy as np
from typing import Dict, Optional, Literal

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KoshiFeedback:
    """Process previous frame for coherent animation with color matching and enhancement."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("enhanced_image", "encoded_latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "vae": ("VAE",),
                "color_match_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "noise_amount": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
                "sharpen_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "contrast_boost": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 1.2, "step": 0.02}),
            },
            "optional": {
                "denoise_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "auto_correct": ("BOOLEAN", {"default": False}),
            }
        }

    def process(
        self,
        current_image: torch.Tensor,
        reference_image: torch.Tensor,
        vae,
        color_match_strength: float,
        noise_amount: float,
        sharpen_amount: float,
        contrast_boost: float,
        denoise_strength: float = 0.65,
        auto_correct: bool = False,
    ):
        """Apply feedback enhancement pipeline."""
        # Convert to numpy for processing (B, H, W, C) -> (H, W, C)
        current_np = (current_image[0].cpu().numpy() * 255).astype(np.uint8)
        reference_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)

        result = current_np.copy()

        # 0. Auto-correction: detect burn/blur and compensate
        if auto_correct:
            if self.detect_burn(result):
                # Burned out frame — blend heavily with reference to recover
                result = (result.astype(np.float32) * 0.3 +
                         reference_np.astype(np.float32) * 0.7).astype(np.uint8)
            if self.detect_blur(result):
                # Blurry frame — force stronger sharpening
                sharpen_amount = max(sharpen_amount, 0.3)

        # 1. Color matching (LAB space histogram matching)
        if color_match_strength > 0:
            result = self._color_match(result, reference_np, color_match_strength)

        # 2. Contrast adjustment
        if contrast_boost != 1.0:
            result = self._apply_contrast(result, contrast_boost)

        # 3. Sharpening (unsharp mask)
        if sharpen_amount > 0 and SCIPY_AVAILABLE:
            result = self._apply_sharpening(result, sharpen_amount)

        # 4. Noise injection (MUST be after color matching)
        if noise_amount > 0:
            result = self._apply_noise(result, noise_amount)

        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        result_tensor = result_tensor.to(current_image.device)

        # Encode to latent
        # ComfyUI VAE expects (B, H, W, C) format
        encoded = vae.encode(result_tensor[:, :, :, :3])

        return (result_tensor, {"samples": encoded})

    def _color_match(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Match color statistics using LAB space."""
        try:
            import cv2
            # Convert to LAB
            src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Match mean and std for each channel
            for i in range(3):
                src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std()
                ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()

                if src_std > 1e-6:
                    src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean

            src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
            matched = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)

            # Blend based on strength
            result = (source.astype(np.float32) * (1 - strength) +
                     matched.astype(np.float32) * strength)
            return np.clip(result, 0, 255).astype(np.uint8)

        except ImportError:
            # Fallback: simple RGB mean matching
            for i in range(3):
                src_mean = source[:, :, i].mean()
                ref_mean = reference[:, :, i].mean()
                diff = (ref_mean - src_mean) * strength
                source[:, :, i] = np.clip(source[:, :, i].astype(np.float32) + diff, 0, 255)
            return source.astype(np.uint8)

    def _apply_contrast(self, image: np.ndarray, boost: float) -> np.ndarray:
        """Apply contrast adjustment around midpoint."""
        img_float = image.astype(np.float32)
        midpoint = 127.5
        adjusted = (img_float - midpoint) * boost + midpoint
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _apply_sharpening(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply unsharp mask sharpening."""
        img_float = image.astype(np.float32)
        blurred = gaussian_filter(img_float, sigma=1.0)
        sharpened = img_float + amount * (img_float - blurred)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_noise(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Add gaussian noise to prevent stagnation."""
        img_float = image.astype(np.float32)
        noise = np.random.randn(*image.shape).astype(np.float32)
        noise_scaled = noise * (amount * 15.0)
        noisy = img_float + noise_scaled
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def detect_burn(self, image: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Detect burned-out image by checking normalized standard deviation.

        A burned image has nearly uniform pixel values (all white, all black,
        or any flat color), resulting in very low standard deviation.

        Args:
            image: uint8 image array (H, W, C)
            threshold: Normalized std threshold (0-1 scale). Below this = burned.

        Returns:
            True if image appears burned out
        """
        std_normalized = image.astype(np.float32).std() / 255.0
        return bool(std_normalized < threshold)

    def detect_blur(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Detect excessively blurred image using gradient variance.

        Uses Laplacian variance when cv2 is available, falls back to
        simple gradient magnitude variance.

        Args:
            image: uint8 image array (H, W, C)
            threshold: Variance threshold. Below this = blurry.

        Returns:
            True if image appears too blurry
        """
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return bool(laplacian_var < threshold)
        except ImportError:
            # Fallback: simple gradient magnitude variance
            gray = image.astype(np.float32).mean(axis=2)
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            gradient_var = (dx.var() + dy.var()) / 2.0
            return bool(gradient_var < threshold)


class KoshiFeedbackSimple:
    """Simplified feedback - just encode previous frame with optional noise."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "noise_amount": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
            }
        }

    def process(self, image: torch.Tensor, vae, noise_amount: float):
        """Encode image to latent with optional noise."""
        if noise_amount > 0:
            noise = torch.randn_like(image) * noise_amount
            image = torch.clamp(image + noise, 0, 1)

        encoded = vae.encode(image[:, :, :, :3])
        return ({"samples": encoded},)


NODE_CLASS_MAPPINGS = {
    "Koshi_Feedback": KoshiFeedback,
    "Koshi_FeedbackSimple": KoshiFeedbackSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_Feedback": "▀▄▀ KN Feedback",
    "Koshi_FeedbackSimple": "▀▄▀ KN Feedback Simple",
}
