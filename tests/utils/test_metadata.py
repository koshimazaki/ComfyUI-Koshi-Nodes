"""Tests for nodes.utils.metadata -- capture and serialization utilities."""

import sys
import os
import json
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nodes.utils.metadata import (
    capture_settings,
    _serialize_value,
    save_metadata,
    load_metadata,
    metadata_to_string,
)


# ---------------------------------------------------------------------------
# capture_settings
# ---------------------------------------------------------------------------

class TestCaptureSettings:
    """Verify that capture_settings builds a well-formed metadata dict."""

    def test_returns_dict_with_timestamp(self):
        result = capture_settings()
        assert isinstance(result, dict)
        assert "timestamp" in result

    def test_timestamp_is_iso_format(self):
        result = capture_settings()
        ts = result["timestamp"]
        # ISO format contains 'T' separator between date and time
        assert "T" in ts

    def test_kwargs_stored_under_settings_key(self):
        result = capture_settings(steps=20, cfg=7.5)
        assert "settings" in result
        assert result["settings"]["steps"] == 20
        assert result["settings"]["cfg"] == 7.5

    def test_empty_kwargs_produces_empty_settings(self):
        result = capture_settings()
        assert result["settings"] == {}


# ---------------------------------------------------------------------------
# _serialize_value
# ---------------------------------------------------------------------------

class TestSerializeValue:
    """Verify JSON-safe conversion for various Python types."""

    def test_torch_tensor(self):
        t = torch.rand(2, 3, 4)
        result = _serialize_value(t)
        assert result["type"] == "tensor"
        assert result["shape"] == [2, 3, 4]
        assert "dtype" in result

    def test_numpy_array(self):
        arr = np.zeros((5, 10), dtype=np.float64)
        result = _serialize_value(arr)
        assert result["type"] == "ndarray"
        assert result["shape"] == [5, 10]
        assert "float64" in result["dtype"]

    def test_list_recursive(self):
        src = [torch.rand(2, 2), 42, "hello"]
        result = _serialize_value(src)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "tensor"
        assert result[1] == 42
        assert result[2] == "hello"

    def test_tuple_recursive(self):
        src = (1, torch.rand(3,))
        result = _serialize_value(src)
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1]["type"] == "tensor"

    def test_dict_recursive(self):
        src = {"a": torch.rand(1), "b": {"nested": np.array([1, 2])}}
        result = _serialize_value(src)
        assert result["a"]["type"] == "tensor"
        assert result["b"]["nested"]["type"] == "ndarray"

    def test_primitive_int(self):
        assert _serialize_value(42) == 42

    def test_primitive_str(self):
        assert _serialize_value("hello") == "hello"

    def test_primitive_float(self):
        assert _serialize_value(3.14) == 3.14

    def test_object_with_dunder_dict(self):
        """Objects with __dict__ should return type + attrs string."""

        class SampleObj:
            def __init__(self):
                self.x = 10

        obj = SampleObj()
        result = _serialize_value(obj)
        assert "type" in result
        assert result["type"] == "SampleObj"
        assert "attrs" in result


# ---------------------------------------------------------------------------
# save_metadata / load_metadata  (filesystem round-trip)
# ---------------------------------------------------------------------------

class TestSaveLoadMetadata:
    """Verify JSON persistence via save and load."""

    def test_round_trip(self, tmp_path):
        original = capture_settings(sampler="euler", steps=30)
        filepath = str(tmp_path / "meta.json")
        returned_path = save_metadata(original, filepath)
        assert returned_path == filepath
        loaded = load_metadata(filepath)
        assert loaded["settings"]["sampler"] == "euler"
        assert loaded["settings"]["steps"] == 30
        assert "timestamp" in loaded

    def test_creates_parent_directories(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "meta.json")
        save_metadata({"test": True}, deep_path)
        assert os.path.isfile(deep_path)

    def test_saved_file_is_valid_json(self, tmp_path):
        filepath = str(tmp_path / "check.json")
        save_metadata({"key": "value"}, filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["key"] == "value"


# ---------------------------------------------------------------------------
# metadata_to_string
# ---------------------------------------------------------------------------

class TestMetadataToString:
    """Verify string serialization of metadata dicts."""

    def test_returns_valid_json_string(self):
        meta = capture_settings(width=512, height=768)
        result = metadata_to_string(meta)
        parsed = json.loads(result)
        assert parsed["settings"]["width"] == 512

    def test_handles_non_serializable_via_default_str(self):
        """default=str fallback should prevent TypeError for exotic types."""
        meta = {"obj": object()}
        result = metadata_to_string(meta)
        # Should not raise; value becomes its str() repr
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Complex / integration-level
# ---------------------------------------------------------------------------

class TestComplexMetadata:
    """End-to-end test with deeply nested, mixed-type metadata."""

    def test_nested_metadata_serializes_without_error(self, tmp_path):
        meta = capture_settings(
            model_config={"layers": [64, 128, 256], "activation": "relu"},
            latent=torch.randn(1, 4, 8, 8),
            mask=np.ones((64, 64), dtype=np.float32),
            prompt="a scenic mountain",
            negative="blurry",
            seed=12345,
        )
        filepath = str(tmp_path / "complex.json")
        save_metadata(meta, filepath)
        loaded = load_metadata(filepath)
        assert loaded["settings"]["seed"] == 12345
        assert loaded["settings"]["latent"]["type"] == "tensor"
        assert loaded["settings"]["mask"]["type"] == "ndarray"
        assert loaded["settings"]["model_config"]["layers"] == [64, 128, 256]
