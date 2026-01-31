"""Shared preview utilities for Koshi nodes."""
import numpy as np
import os
import uuid

try:
    from PIL import Image
    import folder_paths
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False


def save_images_for_preview(image_tensor, prefix="koshi"):
    """
    Save images to temp folder and return preview metadata for ComfyUI UI.

    Args:
        image_tensor: torch.Tensor of shape (B, H, W, C) or (H, W, C)
        prefix: filename prefix for saved images

    Returns:
        List of dicts with filename, subfolder, type for ComfyUI preview
    """
    if not PREVIEW_AVAILABLE:
        return []

    results = []
    output_dir = folder_paths.get_temp_directory()

    # Handle batch
    if len(image_tensor.shape) == 4:
        batch = image_tensor
    else:
        batch = image_tensor.unsqueeze(0)

    for i in range(batch.shape[0]):
        img_np = batch[i].cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_np)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)

        results.append({
            "filename": filename,
            "subfolder": "",
            "type": "temp"
        })

    return results


def make_preview_result(output_tensor, *additional_outputs, prefix="koshi"):
    """
    Create a ComfyUI result dict with preview UI.

    Args:
        output_tensor: The main image output tensor
        *additional_outputs: Any additional outputs to include in result tuple
        prefix: Filename prefix for preview images

    Returns:
        Dict with "ui" and "result" keys for ComfyUI
    """
    preview_images = save_images_for_preview(output_tensor, prefix)
    return {
        "ui": {"images": preview_images},
        "result": (output_tensor,) + additional_outputs
    }
