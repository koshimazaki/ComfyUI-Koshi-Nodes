"""Metadata capture and save nodes."""
import json
import os
from datetime import datetime


class KoshiCaptureSettings:
    """Capture all workflow settings as JSON metadata."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Utility"
    FUNCTION = "capture"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("metadata_json", "workflow_json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "include_workflow": ("BOOLEAN", {"default": True}),
                "include_node_settings": ("BOOLEAN", {"default": True}),
                "pretty_print": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_note": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    def capture(self, include_workflow, include_node_settings, pretty_print,
                image=None, custom_note="", prompt=None, extra_pnginfo=None, unique_id=None):
        
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
        
        # Extract all node settings from workflow prompt
        if include_node_settings and prompt:
            metadata["nodes"] = self._extract_node_settings(prompt)
            metadata["generation_params"] = self._extract_generation_params(prompt)
        
        # Full workflow JSON
        workflow_json = ""
        if include_workflow and extra_pnginfo:
            workflow_data = extra_pnginfo.get("workflow", {})
            workflow_json = json.dumps(workflow_data, indent=2 if pretty_print else None)
        
        indent = 2 if pretty_print else None
        metadata_str = json.dumps(metadata, indent=indent, default=str)
        
        return (metadata_str, workflow_json)
    
    def _extract_node_settings(self, prompt: dict) -> dict:
        """Extract settings from all nodes in the workflow."""
        nodes = {}
        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "Unknown")
            inputs = node_data.get("inputs", {})
            
            # Filter out connection references (lists like [node_id, slot])
            settings = {}
            for key, value in inputs.items():
                if not isinstance(value, list):
                    settings[key] = value
            
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
            
            # CLIP text encode (prompts)
            if "CLIPTextEncode" in class_type:
                if "text" in inputs and not isinstance(inputs["text"], list):
                    if "positive" not in params:
                        params["positive_prompt"] = inputs["text"]
                    else:
                        params["negative_prompt"] = inputs["text"]
            
            # Empty latent (dimensions)
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


class KoshiSaveMetadata:
    """Save metadata JSON to file."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Utility"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata_json": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": "generation_settings"}),
            },
            "optional": {
                "workflow_json": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": ""}),
                "append_timestamp": ("BOOLEAN", {"default": True}),
                "save_workflow_separate": ("BOOLEAN", {"default": False}),
            }
        }

    def save(self, metadata_json, filename, workflow_json="", output_path="", 
             append_timestamp=True, save_workflow_separate=False):
        
        if not output_path:
            output_path = os.path.join(os.path.expanduser("~"), "ComfyUI", "output", "metadata")
        
        os.makedirs(output_path, exist_ok=True)
        
        safe_name = "".join(c for c in filename if c.isalnum() or c in "_-")
        if append_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{safe_name}_{timestamp}"
        
        # Save metadata
        filepath = os.path.join(output_path, f"{safe_name}.json")
        with open(filepath, "w") as f:
            f.write(metadata_json)
        
        # Optionally save workflow separately
        if save_workflow_separate and workflow_json:
            workflow_path = os.path.join(output_path, f"{safe_name}_workflow.json")
            with open(workflow_path, "w") as f:
                f.write(workflow_json)
        
        return (filepath,)


class KoshiDisplayMetadata:
    """Display metadata in UI (no file save)."""
    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Utility"
    FUNCTION = "display"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata_json": ("STRING", {"forceInput": True}),
            }
        }

    def display(self, metadata_json):
        # Just pass through - UI will show it
        return (metadata_json,)
