"""Tests for utility nodes: KoshiCaptureSettings, KoshiSaveMetadata,
KoshiDisplayMetadata."""

import sys
import os
import json

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import pytest
import torch

from nodes.utility.metadata_node import (
    KoshiCaptureSettings,
    KoshiSaveMetadata,
    KoshiDisplayMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(batch=1, h=64, w=64, c=3):
    """Create a ComfyUI-format image tensor [B, H, W, C] in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(batch, h, w, c, dtype=torch.float32)


def _mock_prompt():
    """Return a minimal ComfyUI prompt dict with sampler and model nodes."""
    return {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors",
            },
        },
    }


# ===================================================================
# 1. KoshiCaptureSettings
# ===================================================================

class TestKoshiCaptureSettings:
    """Verify the capture method builds well-formed metadata."""

    def test_returns_valid_json(self):
        """metadata_json output should be parseable JSON."""
        node = KoshiCaptureSettings()
        result = node.capture(
            include_workflow=False,
            include_node_settings=False,
            pretty_print=True,
        )
        metadata_str = result[0]
        parsed = json.loads(metadata_str)
        assert isinstance(parsed, dict)

    def test_contains_timestamp(self):
        """Metadata must contain an ISO-format timestamp."""
        node = KoshiCaptureSettings()
        result = node.capture(
            include_workflow=False,
            include_node_settings=False,
            pretty_print=False,
        )
        parsed = json.loads(result[0])
        assert "timestamp" in parsed
        assert "T" in parsed["timestamp"]

    def test_custom_note_included(self):
        """Custom note should appear in metadata when provided."""
        node = KoshiCaptureSettings()
        result = node.capture(
            include_workflow=False,
            include_node_settings=False,
            pretty_print=True,
            custom_note="My test note",
        )
        parsed = json.loads(result[0])
        assert parsed["note"] == "My test note"

    def test_image_shape_included(self):
        """When image is provided, output_image shape info should be present."""
        node = KoshiCaptureSettings()
        image = _make_image(batch=2, h=128, w=256)
        result = node.capture(
            include_workflow=False,
            include_node_settings=False,
            pretty_print=True,
            image=image,
        )
        parsed = json.loads(result[0])
        assert "output_image" in parsed
        assert parsed["output_image"]["batch_size"] == 2
        assert parsed["output_image"]["height"] == 128
        assert parsed["output_image"]["width"] == 256

    def test_generation_params_from_prompt(self):
        """Node settings extraction should populate generation_params."""
        node = KoshiCaptureSettings()
        prompt = _mock_prompt()
        result = node.capture(
            include_workflow=False,
            include_node_settings=True,
            pretty_print=False,
            prompt=prompt,
        )
        parsed = json.loads(result[0])
        assert "generation_params" in parsed
        params = parsed["generation_params"]
        assert params["seed"] == 42
        assert params["steps"] == 20
        assert params["model"] == "sd_xl_base_1.0.safetensors"


# ===================================================================
# 2. KoshiSaveMetadata
# ===================================================================

class TestKoshiSaveMetadata:
    """Verify file persistence of metadata JSON."""

    def test_creates_json_file(self, tmp_path):
        """save should write a .json file to disk."""
        node = KoshiSaveMetadata()
        metadata_json = json.dumps({"test": "value"})
        result = node.save(
            metadata_json=metadata_json,
            filename="test_meta",
            output_path=str(tmp_path),
            append_timestamp=False,
        )
        file_path = result[0]
        assert os.path.isfile(file_path)
        assert file_path.endswith(".json")

    def test_appends_timestamp_to_filename(self, tmp_path):
        """With append_timestamp=True, filename should contain date string."""
        node = KoshiSaveMetadata()
        metadata_json = json.dumps({"ts": True})
        result = node.save(
            metadata_json=metadata_json,
            filename="stamped",
            output_path=str(tmp_path),
            append_timestamp=True,
        )
        file_path = result[0]
        basename = os.path.basename(file_path)
        # Timestamp format is YYYYMMDD_HHMMSS, so name is longer than "stamped.json"
        assert len(basename) > len("stamped.json")

    def test_returns_valid_file_path(self, tmp_path):
        """Returned path should be an absolute path to an existing file."""
        node = KoshiSaveMetadata()
        metadata_json = json.dumps({"key": 123})
        result = node.save(
            metadata_json=metadata_json,
            filename="path_test",
            output_path=str(tmp_path),
            append_timestamp=False,
        )
        file_path = result[0]
        assert os.path.isabs(file_path)
        assert os.path.exists(file_path)


# ===================================================================
# 3. KoshiDisplayMetadata
# ===================================================================

class TestKoshiDisplayMetadata:
    """Verify passthrough behavior of display node."""

    def test_passthrough(self):
        """Input metadata_json should equal output metadata."""
        node = KoshiDisplayMetadata()
        input_json = json.dumps({"a": 1, "b": "hello"})
        result = node.display(metadata_json=input_json)
        assert result[0] == input_json

    def test_works_with_any_json_string(self):
        """Should accept any valid JSON string without modification."""
        node = KoshiDisplayMetadata()
        cases = [
            '{"nested": {"deep": [1,2,3]}}',
            '"just a string"',
            '42',
            'null',
        ]
        for case in cases:
            result = node.display(metadata_json=case)
            assert result[0] == case
