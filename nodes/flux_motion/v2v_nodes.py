"""Koshi V2V Nodes - Only the unique pieces not in standard ComfyUI."""

import torch
import numpy as np
import warnings

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class KoshiColorMatchLAB:
    """Match colors to anchor frame using LAB space - critical for V2V coherence."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "match"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "anchor": ("IMAGE",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def match(self, image: torch.Tensor, anchor: torch.Tensor, strength: float = 1.0):
        """Match image colors to anchor using LAB statistics."""
        if not CV2_AVAILABLE or strength == 0:
            return (image,)

        batch_size = image.shape[0]
        results = []

        # Use first anchor frame for all
        anc_np = (anchor[0].cpu().numpy() * 255).astype(np.uint8)
        anc_lab = cv2.cvtColor(anc_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        anc_stats = [(anc_lab[:,:,i].mean(), anc_lab[:,:,i].std() + 1e-6) for i in range(3)]

        for b in range(batch_size):
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            for i in range(3):
                i_mean, i_std = img_lab[:,:,i].mean(), img_lab[:,:,i].std() + 1e-6
                a_mean, a_std = anc_stats[i]
                img_lab[:,:,i] = (img_lab[:,:,i] - i_mean) * (a_std / i_std) + a_mean

            matched = cv2.cvtColor(np.clip(img_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

            if strength < 1.0:
                matched = ((1 - strength) * img_np + strength * matched).astype(np.uint8)

            results.append(torch.from_numpy(matched.astype(np.float32) / 255.0))

        return (torch.stack(results).to(image.device),)


class KoshiOpticalFlowWarp:
    """Warp image using optical flow from frame pair - for motion transfer.
    
    Supports batch processing: computes flow between corresponding frames
    in flow_from and flow_to, then warps each frame in image_to_warp.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "warp"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_to_warp": ("IMAGE",),
                "flow_from": ("IMAGE",),
                "flow_to": ("IMAGE",),
            },
            "optional": {
                "method": (["dis", "farneback"],),
            }
        }

    def _compute_flow(self, from_img: np.ndarray, to_img: np.ndarray, method: str) -> np.ndarray:
        """Compute optical flow between two frames."""
        from_gray = cv2.cvtColor(from_img, cv2.COLOR_RGB2GRAY)
        to_gray = cv2.cvtColor(to_img, cv2.COLOR_RGB2GRAY)

        if method == "dis":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(from_gray, to_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                from_gray, to_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        return flow

    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp an image using optical flow."""
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped = cv2.remap(
            image,
            x + flow[:, :, 0],
            y + flow[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return warped

    def warp(
        self,
        image_to_warp: torch.Tensor,
        flow_from: torch.Tensor,
        flow_to: torch.Tensor,
        method: str = "dis"
    ):
        """Compute flow between frames and warp images - supports batch processing."""
        if not CV2_AVAILABLE:
            return (image_to_warp,)

        batch_size = image_to_warp.shape[0]
        flow_from_size = flow_from.shape[0]
        flow_to_size = flow_to.shape[0]
        
        results = []

        for b in range(batch_size):
            # Get corresponding flow frame indices (cycle if needed)
            from_idx = min(b, flow_from_size - 1)
            to_idx = min(b, flow_to_size - 1)
            
            # Convert to numpy
            from_np = (flow_from[from_idx].cpu().numpy() * 255).astype(np.uint8)
            to_np = (flow_to[to_idx].cpu().numpy() * 255).astype(np.uint8)
            img_np = (image_to_warp[b].cpu().numpy() * 255).astype(np.uint8)
            
            # Compute flow and warp
            flow = self._compute_flow(from_np, to_np, method)
            warped = self._warp_image(img_np, flow)
            
            results.append(torch.from_numpy(warped.astype(np.float32) / 255.0))

        result_tensor = torch.stack(results).to(image_to_warp.device)
        return (result_tensor,)


class KoshiImageBlend:
    """Alpha blend two images together - essential utility for V2V pipelines.
    
    Supports:
    - Constant alpha blending: result = image1 * (1 - alpha) + image2 * alpha
    - Mask-based blending: per-pixel alpha from mask input
    - Batch processing: blends corresponding frames
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "blend"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def blend(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        alpha: float,
        mask: torch.Tensor = None
    ):
        """Blend two images with alpha or mask."""
        # Handle batch size mismatch by broadcasting
        batch1 = image1.shape[0]
        batch2 = image2.shape[0]
        max_batch = max(batch1, batch2)
        
        results = []
        
        for b in range(max_batch):
            idx1 = min(b, batch1 - 1)
            idx2 = min(b, batch2 - 1)
            
            img1 = image1[idx1]
            img2 = image2[idx2]
            
            if mask is not None:
                # Use mask for per-pixel alpha
                mask_idx = min(b, mask.shape[0] - 1) if mask.dim() >= 1 else 0
                m = mask[mask_idx] if mask.dim() >= 1 else mask
                
                # Expand mask to match image dimensions (H, W) -> (H, W, C)
                if m.dim() == 2:
                    m = m.unsqueeze(-1).expand_as(img1)
                elif m.dim() == 3 and m.shape[-1] == 1:
                    m = m.expand_as(img1)
                
                # Ensure mask is on same device
                m = m.to(img1.device)
                
                blended = img1 * (1 - m) + img2 * m
            else:
                blended = img1 * (1 - alpha) + img2 * alpha
            
            results.append(blended)
        
        return (torch.stack(results).to(image1.device),)


class KoshiV2VProcessor:
    """
    [DEPRECATED] Use external KSampler nodes instead.
    
    This node uses internal ComfyUI sampling APIs that are not stable.
    For V2V workflows, connect nodes like this:
    
    LoadVideo -> KoshiColorMatchLAB -> VAE Encode -> KSampler -> VAE Decode -> KoshiImageBlend
    
    The modular approach gives you more control and works with FLUX/other models.
    """
    COLOR = "#4a1a1a"  # Reddish to indicate deprecated
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "mode": (["pure", "temporal", "motion", "ultimate"],),
                "denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "denoise_first": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "temporal_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "flow_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "ref_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            }
        }

    def process(
        self,
        images: torch.Tensor,
        model,
        positive,
        negative,
        vae,
        mode: str,
        denoise: float,
        steps: int,
        cfg: float,
        seed: int,
        reference_image: torch.Tensor = None,
        denoise_first: float = 0.7,
        temporal_blend: float = 0.3,
        flow_blend: float = 0.5,
        color_match: float = 1.0,
        ref_blend: float = 0.3,
    ):
        """[DEPRECATED] This node is deprecated. Use modular nodes with external KSampler."""
        warnings.warn(
            "KoshiV2VProcessor is DEPRECATED. Use KoshiColorMatchLAB + external KSampler + "
            "KoshiImageBlend for V2V workflows. This node uses unstable internal APIs.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return input unchanged with deprecation notice
        # The old implementation used comfy.sample.sample() which is not a stable API
        print("[Koshi] WARNING: KoshiV2VProcessor is deprecated and disabled.")
        print("[Koshi] Use modular workflow: LoadVideo -> KoshiColorMatchLAB -> VAE Encode -> KSampler -> VAE Decode")
        return (images,)


class KoshiV2VMetadata:
    """Save V2V processing metadata alongside video output."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Motion"
    FUNCTION = "save"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ("STRING", {"default": "v2v_temporal"}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "forceInput": True}),
                "mode": ("STRING", {"default": ""}),
                "denoise": ("FLOAT", {"default": 0.65}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 3.5}),
                "seed": ("INT", {"default": 0}),
                "frame_count": ("INT", {"default": 0}),
                "output_path": ("STRING", {"default": ""}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    def save(self, preset, prompt="", mode="", denoise=0.65, steps=20, cfg=3.5,
             seed=0, frame_count=0, output_path="", extra_pnginfo=None):
        """Save V2V metadata JSON."""
        import json
        import os
        from datetime import datetime

        metadata = {
            "generator": "Koshi V2V",
            "preset": preset,
            "timestamp": datetime.now().isoformat(),
            "params": {
                "mode": mode,
                "denoise": denoise,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "frame_count": frame_count,
            }
        }

        if prompt:
            metadata["prompt"] = prompt

        if not output_path:
            output_path = os.path.join(os.path.expanduser("~"), "ComfyUI", "output")

        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_path, f"koshi_v2v_{preset}_{timestamp}.json")

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"ui": {"text": [f"Saved: {filepath}"]}}


NODE_CLASS_MAPPINGS = {
    "Koshi_ColorMatchLAB": KoshiColorMatchLAB,
    "Koshi_OpticalFlowWarp": KoshiOpticalFlowWarp,
    "Koshi_ImageBlend": KoshiImageBlend,
    "Koshi_V2VMetadata": KoshiV2VMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_ColorMatchLAB": "▄▀▄ KN Color Match LAB",
    "Koshi_OpticalFlowWarp": "▄▀▄ KN Optical Flow Warp",
    "Koshi_ImageBlend": "▄▀▄ KN Image Blend",
    "Koshi_V2VMetadata": "▄▀▄ KN V2V Metadata",
}
