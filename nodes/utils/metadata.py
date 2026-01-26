"""Metadata utilities for capturing and saving generation settings."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def capture_settings(**kwargs) -> dict:
    """Capture node settings as a dictionary."""
    return {
        "timestamp": datetime.now().isoformat(),
        "settings": {k: _serialize_value(v) for k, v in kwargs.items()},
    }


def _serialize_value(value: Any) -> Any:
    """Convert value to JSON-serializable format."""
    import torch
    import numpy as np
    
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    elif isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif hasattr(value, "__dict__"):
        return {"type": type(value).__name__, "attrs": str(value)}
    return value


def save_metadata(metadata: dict, filepath: str) -> str:
    """Save metadata to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return filepath


def load_metadata(filepath: str) -> dict:
    """Load metadata from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def metadata_to_string(metadata: dict) -> str:
    """Convert metadata to formatted string."""
    return json.dumps(metadata, indent=2, default=str)
