"""
Koshi Metadata - Unified metadata capture, display, and save.
Combines all metadata functionality into one node.
"""
import json
import os
from datetime import datetime


class KoshiMetadata:
    """
    Unified metadata node - captures workflow settings, displays them, and optionally saves.
    Combines CaptureSettings + DisplayMetadata + SaveMetadata into one.
    """
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Utility"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metadata_json", "display_text", "file_path")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["capture_only", "capture_and_save", "save_json", "save_text"],
                           {"default": "capture_only"}),
            },
            "optional": {
                # Capture options
                "image": ("IMAGE",),
                "custom_note": ("STRING", {"default": "", "multiline": True}),
                "include_workflow": ("BOOLEAN", {"default": True}),
                "include_node_settings": ("BOOLEAN", {"default": True}),

                # Save options
                "filename": ("STRING", {"default": "metadata"}),
                "output_path": ("STRING", {"default": ""}),
                "append_timestamp": ("BOOLEAN", {"default": True}),

                # Input for save_json/save_text modes
                "input_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    def _extract_node_settings(self, prompt: dict) -> dict:
        """Extract settings from all nodes in the workflow."""
        nodes = {}
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "Unknown")
            inputs = node_data.get("inputs", {})

            settings = {k: v for k, v in inputs.items() if not isinstance(v, list)}
            if settings:
                nodes[f"{class_type}_{node_id}"] = settings

        return nodes

    def _extract_generation_params(self, prompt: dict) -> dict:
        """Extract common generation parameters."""
        params = {}

        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})

            # KSampler params
            if "Sampler" in class_type or "sampler" in class_type.lower():
                for key in ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]:
                    if key in inputs and not isinstance(inputs[key], list):
                        params[key] = inputs[key]

            # Model loader
            if "Loader" in class_type or "CheckpointLoader" in class_type:
                if "ckpt_name" in inputs:
                    params["model"] = inputs["ckpt_name"]

            # CLIP text encode
            if "CLIPTextEncode" in class_type:
                if "text" in inputs and not isinstance(inputs["text"], list):
                    if "positive_prompt" not in params:
                        params["positive_prompt"] = inputs["text"]
                    else:
                        params["negative_prompt"] = inputs["text"]

            # Empty latent
            if "EmptyLatent" in class_type:
                for key in ["width", "height", "batch_size"]:
                    if key in inputs:
                        params[key] = inputs[key]

            # LoRA
            if "LoraLoader" in class_type:
                if "lora_name" in inputs:
                    if "loras" not in params:
                        params["loras"] = []
                    params["loras"].append({
                        "name": inputs.get("lora_name"),
                        "strength_model": inputs.get("strength_model"),
                        "strength_clip": inputs.get("strength_clip"),
                    })

        return params

    def _format_display(self, metadata: dict) -> str:
        """Format metadata for human-readable display."""
        lines = ["=== Generation Metadata ===", ""]

        if "timestamp" in metadata:
            lines.append(f"Time: {metadata['timestamp']}")

        if "note" in metadata:
            lines.append(f"Note: {metadata['note']}")
            lines.append("")

        if "output_image" in metadata:
            img = metadata["output_image"]
            lines.append(f"Output: {img['width']}x{img['height']} (batch: {img['batch_size']})")
            lines.append("")

        if "generation_params" in metadata:
            params = metadata["generation_params"]
            lines.append("--- Generation Params ---")
            if "model" in params:
                lines.append(f"Model: {params['model']}")
            if "seed" in params:
                lines.append(f"Seed: {params['seed']}")
            if "steps" in params:
                lines.append(f"Steps: {params['steps']}")
            if "cfg" in params:
                lines.append(f"CFG: {params['cfg']}")
            if "sampler_name" in params:
                lines.append(f"Sampler: {params['sampler_name']}")
            if "positive_prompt" in params:
                prompt = params["positive_prompt"][:100] + "..." if len(params["positive_prompt"]) > 100 else params["positive_prompt"]
                lines.append(f"Prompt: {prompt}")
            lines.append("")

        return "\n".join(lines)

    def _save_to_file(self, content: str, filename: str, output_path: str,
                      append_timestamp: bool, extension: str) -> str:
        """Save content to file."""
        if not output_path:
            output_path = os.path.join(os.path.expanduser("~"), "ComfyUI", "output", "metadata")
        os.makedirs(output_path, exist_ok=True)

        safe_name = "".join(c for c in filename if c.isalnum() or c in "_-") or "metadata"
        if append_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{safe_name}_{timestamp}"

        filepath = os.path.join(output_path, f"{safe_name}.{extension}")
        with open(filepath, "w") as f:
            f.write(content)

        return filepath

    def process(self, action, image=None, custom_note="", include_workflow=True,
                include_node_settings=True, filename="metadata", output_path="",
                append_timestamp=True, input_json="", prompt=None, extra_pnginfo=None):

        metadata = {}
        metadata_json = ""
        display_text = ""
        file_path = ""

        # Capture mode
        if action in ["capture_only", "capture_and_save"]:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "generator": "Koshi-Nodes",
            }

            if custom_note:
                metadata["note"] = custom_note

            if image is not None:
                metadata["output_image"] = {
                    "shape": list(image.shape),
                    "batch_size": image.shape[0],
                    "height": image.shape[1],
                    "width": image.shape[2],
                }

            if include_node_settings and prompt:
                metadata["nodes"] = self._extract_node_settings(prompt)
                metadata["generation_params"] = self._extract_generation_params(prompt)

            if include_workflow and extra_pnginfo:
                metadata["workflow"] = extra_pnginfo.get("workflow", {})

            metadata_json = json.dumps(metadata, indent=2, default=str)
            display_text = self._format_display(metadata)

            if action == "capture_and_save":
                file_path = self._save_to_file(metadata_json, filename, output_path,
                                                append_timestamp, "json")

        # Save JSON input
        elif action == "save_json":
            metadata_json = input_json
            display_text = input_json[:500] + "..." if len(input_json) > 500 else input_json
            file_path = self._save_to_file(input_json, filename, output_path,
                                            append_timestamp, "json")

        # Save as text
        elif action == "save_text":
            content = input_json if input_json else display_text
            metadata_json = content
            display_text = content
            file_path = self._save_to_file(content, filename, output_path,
                                            append_timestamp, "txt")

        return (metadata_json, display_text, file_path)


NODE_CLASS_MAPPINGS = {"Koshi_Metadata": KoshiMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"Koshi_Metadata": "◊ Koshi Metadata"}
