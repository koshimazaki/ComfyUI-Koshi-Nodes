"""Koshi V2V Nodes - Only the unique pieces not in standard ComfyUI."""

import torch
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class KoshiColorMatchLAB:
    """Match colors to anchor frame using LAB space - critical for V2V coherence."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/V2V"
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
    """Warp image using optical flow from frame pair - for motion transfer."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/V2V"
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

    def warp(
        self,
        image_to_warp: torch.Tensor,
        flow_from: torch.Tensor,
        flow_to: torch.Tensor,
        method: str = "dis"
    ):
        """Compute flow between frames and warp image."""
        if not CV2_AVAILABLE:
            return (image_to_warp,)

        # Get flow
        from_gray = cv2.cvtColor(
            (flow_from[0].cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )
        to_gray = cv2.cvtColor(
            (flow_to[0].cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        if method == "dis":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(from_gray, to_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                from_gray, to_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        # Warp
        img_np = (image_to_warp[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        warped = cv2.remap(
            img_np,
            x + flow[:, :, 0],
            y + flow[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        result = torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0)
        return (result.to(image_to_warp.device),)


class KoshiV2VProcessor:
    """
    Main V2V processor - handles temporal coherence for video stylization.

    Modes:
    - pure: Frame-by-frame with color matching only
    - temporal: Blend previous output with current input
    - motion: Warp previous output using optical flow
    - ultimate: Motion + temporal + init image (all techniques combined)
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/V2V"
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
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
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

    def _warp_with_flow(self, image: torch.Tensor, flow_from: torch.Tensor, flow_to: torch.Tensor) -> torch.Tensor:
        """Warp image using optical flow between two frames."""
        from_gray = cv2.cvtColor(
            (flow_from[0].cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )
        to_gray = cv2.cvtColor(
            (flow_to[0].cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(from_gray, to_gray, None)

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped = cv2.remap(img_np, x + flow[:, :, 0], y + flow[:, :, 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        result = torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0)
        return result.to(image.device)

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
        """Process video frames with V2V pipeline."""
        import comfy.sample

        batch_size = images.shape[0]
        outputs = []
        anchor = None
        prev_output = None
        prev_input = None

        for i in range(batch_size):
            frame = images[i:i+1]
            current_denoise = denoise_first if i == 0 else denoise

            # Prepare source based on mode
            if i == 0:
                # First frame - optionally blend with reference image
                if reference_image is not None and ref_blend > 0:
                    source = (1 - ref_blend) * frame + ref_blend * reference_image[:1]
                else:
                    source = frame

            elif mode == "temporal" and prev_output is not None:
                # Blend previous output with current input
                source = (1 - temporal_blend) * prev_output + temporal_blend * frame

            elif mode == "motion" and prev_output is not None and CV2_AVAILABLE:
                # Warp previous output using flow only
                source = self._warp_with_flow(prev_output, prev_input, frame)

            elif mode == "ultimate" and prev_output is not None and CV2_AVAILABLE:
                # Ultimate: motion + temporal combined
                # 1. Warp prev_output using optical flow
                warped = self._warp_with_flow(prev_output, prev_input, frame)
                # 2. Blend warped with current frame (motion guided)
                motion_guided = (1 - flow_blend) * warped + flow_blend * frame
                # 3. Blend with prev_output (temporal)
                if temporal_blend > 0:
                    source = (1 - temporal_blend) * prev_output + temporal_blend * motion_guided
                else:
                    source = motion_guided

            else:
                source = frame

            # Encode
            latent = vae.encode(source[:, :, :, :3])

            # Sample
            samples = comfy.sample.sample(
                model,
                noise=comfy.sample.prepare_noise(latent, seed + i, None),
                steps=steps,
                cfg=cfg,
                sampler_name="euler",
                scheduler="normal",
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=current_denoise,
            )

            # Decode
            output = vae.decode(samples)

            # Color match to anchor
            if i == 0:
                anchor = output
            elif color_match > 0 and CV2_AVAILABLE:
                out_np = (output[0].cpu().numpy() * 255).astype(np.uint8)
                anc_np = (anchor[0].cpu().numpy() * 255).astype(np.uint8)

                out_lab = cv2.cvtColor(out_np, cv2.COLOR_RGB2LAB).astype(np.float32)
                anc_lab = cv2.cvtColor(anc_np, cv2.COLOR_RGB2LAB).astype(np.float32)

                for c in range(3):
                    o_mean, o_std = out_lab[:,:,c].mean(), out_lab[:,:,c].std() + 1e-6
                    a_mean, a_std = anc_lab[:,:,c].mean(), anc_lab[:,:,c].std() + 1e-6
                    out_lab[:,:,c] = (out_lab[:,:,c] - o_mean) * (a_std / o_std) + a_mean

                matched = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                if color_match < 1.0:
                    matched = ((1-color_match) * out_np + color_match * matched).astype(np.uint8)
                output = torch.from_numpy(matched.astype(np.float32) / 255.0).unsqueeze(0)
                output = output.to(frame.device)

            outputs.append(output)
            prev_output = output
            prev_input = frame

            if i % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return (torch.cat(outputs, dim=0),)


class KoshiV2VMetadata:
    """Save V2V processing metadata alongside video output."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/V2V"
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
    "Koshi_V2VProcessor": KoshiV2VProcessor,
    "Koshi_V2VMetadata": KoshiV2VMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_ColorMatchLAB": "▄▀▄ KN Color Match LAB",
    "Koshi_OpticalFlowWarp": "▄▀▄ KN Optical Flow Warp",
    "Koshi_V2VProcessor": "▄▀▄ KN V2V Processor",
    "Koshi_V2VMetadata": "▄▀▄ KN V2V Metadata",
}
