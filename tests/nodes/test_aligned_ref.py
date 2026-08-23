"""AKUSPACE Reference Audio (aligned) — contract tests.

The node's real work happens inside ComfyUI (it monkeypatches
`LTXAVModel._process_input`), which is not importable here. What IS testable off
ComfyUI, and what actually breaks workflows when it drifts, is the contract the
workflow JSON is written against: the node id, the socket names and their order,
the return signature, and the fact that the module imports cleanly without
ComfyUI so the pack still loads.
"""

import pytest

from nodes.audio import aligned_ref
from nodes.audio.aligned_ref import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AKUSPACEReferenceAudioAligned,
)


def test_node_id_matches_the_session_kit_copy():
    """Graphs built against the standalone kit node must resolve against the pack."""

    assert set(NODE_CLASS_MAPPINGS) == {"AKUSPACEReferenceAudioAligned"}
    assert NODE_CLASS_MAPPINGS["AKUSPACEReferenceAudioAligned"] is AKUSPACEReferenceAudioAligned
    assert NODE_DISPLAY_NAME_MAPPINGS["AKUSPACEReferenceAudioAligned"].startswith("◉ AKUSPACE")


def test_comfy_interface():
    assert AKUSPACEReferenceAudioAligned.CATEGORY == "Koshi/Space"
    assert AKUSPACEReferenceAudioAligned.FUNCTION == "apply"
    assert AKUSPACEReferenceAudioAligned.RETURN_TYPES == ("MODEL", "CONDITIONING", "CONDITIONING")
    assert AKUSPACEReferenceAudioAligned.RETURN_NAMES == ("model", "positive", "negative")


def test_socket_names_and_order_match_the_stock_reference_node():
    """The A/B swaps this node for core LTXVReferenceAudio; the shared sockets
    must line up so only the node type changes between the two workflows."""

    required = AKUSPACEReferenceAudioAligned.INPUT_TYPES()["required"]
    assert list(required)[:5] == [
        "model",
        "positive",
        "negative",
        "reference_audio",
        "audio_vae",
    ]
    assert [required[k][0] for k in list(required)[:5]] == [
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "AUDIO",
        "VAE",
    ]


def test_reference_guidance_defaults_to_off():
    """Trainer inference had no guidance term; 0.0 is the faithful default."""

    spec = AKUSPACEReferenceAudioAligned.INPUT_TYPES()["required"]["reference_guidance_scale"]
    assert spec[0] == "FLOAT"
    assert spec[1]["default"] == 0.0


def test_module_imports_without_comfyui_and_says_so():
    """The pack must load outside ComfyUI; the node then fails loudly on use."""

    if aligned_ref._COMFY:  # pragma: no cover - only inside a real ComfyUI
        pytest.skip("ComfyUI core present; the no-core path cannot be exercised")
    with pytest.raises(RuntimeError, match="needs a ComfyUI core"):
        AKUSPACEReferenceAudioAligned().apply(
            model=None,
            positive=None,
            negative=None,
            reference_audio=None,
            audio_vae=None,
            reference_guidance_scale=0.0,
        )


def test_flag_name_is_the_one_the_patch_looks_for():
    """The dict key is the whole contract between apply() and the patch."""

    assert aligned_ref._FLAG == "akuspace_aligned"
