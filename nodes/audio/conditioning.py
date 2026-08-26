"""Pure conditioning logic shared by the Comfy node and its tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PRESET_PATH = Path(__file__).resolve().parent / "presets.json"
PRESET_DATA: dict[str, Any] = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
PRESETS: dict[str, dict[str, Any]] = PRESET_DATA["presets"]
LEVELS: dict[str, dict[str, Any]] = PRESET_DATA["levels"]
SOURCE_TYPES: list[dict[str, str]] = PRESET_DATA["source_types"]
SOURCE_VALUES = [item["value"] for item in SOURCE_TYPES]
CONTROL_SCHEMA: dict[str, Any] = PRESET_DATA["control_schema"]
MODEL_PROFILES: list[dict[str, Any]] = PRESET_DATA["model_profiles"]
DEFAULT_MODEL_PROFILE: str = PRESET_DATA["default_model_profile"]
ACTIVE_MODEL_PROFILE = next(
    (profile for profile in MODEL_PROFILES if profile["id"] == DEFAULT_MODEL_PROFILE),
    MODEL_PROFILES[0],
)
TRIGGER: str = ACTIVE_MODEL_PROFILE.get("trigger", PRESET_DATA["trigger"])

SPACE_MODES = [item["value"] for item in CONTROL_SCHEMA["modes"]]
ROOM_PRESETS = list(CONTROL_SCHEMA["room_presets"])
OUTDOOR_TIMES = list(CONTROL_SCHEMA["outdoor_times"])
OUTDOOR_LEVEL_DEFAULT = CONTROL_SCHEMA["outdoor_level"]
# .get() so an older presets.json without the key still loads.
OUTDOOR_LEVELS = list(CONTROL_SCHEMA.get("outdoor_levels", [OUTDOOR_LEVEL_DEFAULT]))
SFX_PRESETS = list(CONTROL_SCHEMA["sfx_presets"])
SFX_LEVELS = list(CONTROL_SCHEMA["sfx_levels"])
EFFECT_LEVELS = list(CONTROL_SCHEMA["reverb_levels"])


def resolve_preset_key(
    space_mode: str,
    room_preset: str,
    outdoor_time: str,
    sfx_preset: str = "dual_delay",
) -> str:
    """Resolve the active preset without making hidden frontend state authoritative."""

    if space_mode == "dry":
        return "dry"
    if space_mode == "outside":
        return "outdoor_night" if outdoor_time == "night" else "outdoor_day"
    if space_mode == "sfx":
        return sfx_preset if sfx_preset in SFX_PRESETS else "dual_delay"
    return room_preset if room_preset in ROOM_PRESETS else "medium_room"


def source_coverage(source_type: str) -> str:
    for item in SOURCE_TYPES:
        if item["value"] == source_type:
            return item["coverage"]
    return "experimental"


def resolve_level_key(
    space_mode: str,
    effect_level: str,
    sfx_level: str = "low",
) -> str:
    """Resolve only level values represented by the v0.5 training manifest."""

    if space_mode == "outside":
        # Outdoor trained BOTH gentle and heavy — outdoor_day_birds/{low,high}
        # and outdoor_night/{low,high}, 14 rows and a separate audio dir each —
        # but no "mid" cell. Clamp to the trained pair rather than pinning every
        # outdoor request to one level, which made the other unreachable.
        return effect_level if effect_level in OUTDOOR_LEVELS else OUTDOOR_LEVEL_DEFAULT
    if space_mode == "sfx":
        return sfx_level if sfx_level in SFX_LEVELS else SFX_LEVELS[0]
    return effect_level if effect_level in EFFECT_LEVELS else "mid"


def build_caption(
    space_mode: str,
    room_preset: str,
    outdoor_time: str,
    effect_level: str,
    source_type: str = "",
    sfx_preset: str = "dual_delay",
    sfx_level: str = "low",
) -> str:
    """Emit the v0.5 training caption grammar, or a dry bypass instruction."""

    preset_key = resolve_preset_key(space_mode, room_preset, outdoor_time, sfx_preset)
    preset = PRESETS[preset_key]
    subject = f" {source_type.strip()}" if source_type.strip() else ""
    if preset_key == "dry":
        return f"{source_type.strip() + ', ' if source_type.strip() else ''}close-miked dry reference, no reverb, no background ambience"

    level = LEVELS[resolve_level_key(preset["mode"], effect_level, sfx_level)]
    caption = (
        f"{TRIGGER}{subject} {preset['caption_where']}, "
        f"{level['caption_word']} {preset['caption_character']}"
    )
    if preset.get("caption_tail"):
        caption += f", {preset['caption_tail']}"
    return caption


def compose_prompt(base_prompt: str, treatment_prompt: str, enabled: bool = True) -> str:
    """Append AKUSPACE conditioning without replacing the workflow's visual prompt."""

    base = base_prompt.strip()
    treatment = treatment_prompt.strip()
    if not enabled:
        return base
    if not base:
        return treatment
    if not treatment:
        return base
    separator = " " if base.endswith((",", ".", ";", ":")) else ", "
    return f"{base}{separator}{treatment}"


def encode_conditioning(clip, prompt: str):
    """Encode text using the same contract as Comfy's native CLIP Text Encode node."""

    if clip is None:
        raise RuntimeError(
            "ERROR: clip input is invalid: None\n\n"
            "Connect the CLIP output from the checkpoint or text-encoder loader."
        )
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def build_scene(
    space_mode: str,
    room_preset: str,
    outdoor_time: str,
    effect_level: str,
    source_type: str = "",
    sfx_preset: str = "dual_delay",
    sfx_level: str = "low",
) -> dict[str, Any]:
    """Build deterministic metadata for API workflows and the graphical widget."""

    preset_key = resolve_preset_key(space_mode, room_preset, outdoor_time, sfx_preset)
    preset = PRESETS[preset_key]
    active_level_key = resolve_level_key(preset["mode"], effect_level, sfx_level)
    level = LEVELS[active_level_key]
    caption = build_caption(
        space_mode,
        room_preset,
        outdoor_time,
        effect_level,
        source_type,
        sfx_preset,
        sfx_level,
    )
    is_dry = preset_key == "dry"
    coverage = source_coverage(source_type) if source_type else "reference_audio"

    return {
        "schema": "akuspace/spatial-control/v4",
        "release_version": PRESET_DATA["release"]["display_version"],
        "release_stage": PRESET_DATA["release"]["stage"],
        "model_profile": ACTIVE_MODEL_PROFILE["id"],
        "trigger": None if is_dry else TRIGGER,
        "space_mode": preset["mode"],
        "preset_key": preset_key,
        "preset_label": preset["label"],
        "acoustic_fingerprint": preset["acoustic_fingerprint"],
        "effect_level": None if is_dry else active_level_key,
        "effect_relative_db": None if is_dry else level["relative_db"],
        "outdoor_time": preset.get("time_of_day"),
        "sfx_preset": preset_key if preset["mode"] == "sfx" else None,
        "sfx_coverage": preset.get("coverage") if preset["mode"] == "sfx" else None,
        "source_type": source_type or None,
        "source_coverage": coverage,
        "conditioning_status": (
            "bypass"
            if is_dry
            else "experimental_sfx"
            if preset["mode"] == "sfx"
            else "experimental_source"
            if source_type and coverage != "trained"
            else "trained_caption"
        ),
        "recommended_lora_strength": 0.0 if is_dry else 1.0,
        "estimates": {
            "rt60_seconds": preset["estimated_rt60"],
            "predelay_ms": preset["estimated_predelay_ms"],
        },
        "caption": caption,
        "notes": {
            "geometry": "relative visualization only; no room measurements are inferred or exported",
            "level": (
                "trained categorical level; relative_db is a legacy UI proxy and not the "
                "per-treatment v0.5 render gain"
            ),
            "controls": (
                "reverb exposes size plus low/moderate/heavy dry-wet categories; "
                "space exposes day/night; SFX exposes low/high"
            ),
            "sfx": (
                "human label: Dual Delay; trained caption label: modular granular delay"
            ),
        },
    }


def validate_choice(value: str, allowed: list[str], label: str) -> str | None:
    if value not in allowed:
        return f"Unknown {label}: {value}"
    return None
