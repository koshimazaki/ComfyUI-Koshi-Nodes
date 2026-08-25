"""AKUSPACE prompt and text-encoding controls for the spatial-audio LoRA."""

import logging

from .conditioning import (
    EFFECT_LEVELS,
    OUTDOOR_TIMES,
    ROOM_PRESETS,
    SFX_LEVELS,
    SOURCE_VALUES,
    build_caption,
    compose_prompt,
    encode_conditioning,
    validate_choice,
)


logger = logging.getLogger(__name__)

MODE_OPTIONS = ["Off", "Room", "Space", "Sound effects"]
APPLICATION_OPTIONS = ["Off", "Low", "Moderate", "Heavy", "Day", "Night", "High"]
# The caption grammar starts with what the dry recording IS:
# "AKUSPACE female spoken voice in a small bathroom-like room, ...". The node
# shipped without that word (effect-only captions); `source_type` restores it as
# an OPTIONAL control so existing graphs keep their widget layout. "none" keeps
# the effect-only caption.
SOURCE_NONE = "none"
SOURCE_OPTIONS = list(SOURCE_VALUES)
MODE_VALUES = {
    "Off": "dry",
    "Room": "room",
    "Space": "outside",
    "Sound effects": "sfx",
    "dry": "dry",
    "room": "room",
    "outside": "outside",
    "sfx": "sfx",
}
ROOM_APPLICATION = {"Low": "low", "Moderate": "mid", "Heavy": "high"}
SPACE_APPLICATION = {"Day": "day", "Night": "night"}
SFX_APPLICATION = {"Low": "low", "High": "high"}

# Application is one flat combo shared by every mode, so only 8 of the 28
# mode x application pairs mean anything. The graph UI keeps the pair
# consistent; API and headless callers get no such help, so map each mode to
# the applications it actually understands and report the rest.
APPLICATION_BY_MODE = {
    "room": ROOM_APPLICATION,
    "outside": SPACE_APPLICATION,
    "sfx": SFX_APPLICATION,
}
# "Off" means "no Application override, use the mode's own level widget".
# It is not a bypass -- only space_mode="Off" bypasses conditioning.
APPLICATION_NEUTRAL = "Off"


def _controls():
    return {
        "space_mode": (MODE_OPTIONS, {"default": "Room", "display_name": "Mode"}),
        "application": (
            APPLICATION_OPTIONS,
            {"default": "Moderate", "display_name": "Application"},
        ),
        "room_preset": (
            ROOM_PRESETS,
            {"default": "medium_room", "display_name": "Reverb size"},
        ),
        "effect_level": (
            EFFECT_LEVELS,
            {"default": "mid", "display_name": "Dry / wet"},
        ),
        "outdoor_time": (
            OUTDOOR_TIMES,
            {"default": "day", "display_name": "Space"},
        ),
        "sfx_level": (
            SFX_LEVELS,
            {"default": "low", "display_name": "Dry / wet"},
        ),
    }


def _source_control():
    return {
        "source_type": (
            SOURCE_OPTIONS,
            {
                "default": SOURCE_OPTIONS[0],
                "display_name": "Source",
                "tooltip": (
                    "What the dry recording is. Captions begin with it "
                    "('AKUSPACE female spoken voice in ...')."
                ),
            },
        ),
    }


def _normalise_source(source_type):
    if source_type is None:
        return ""
    source_type = str(source_type).strip()
    return "" if source_type == SOURCE_NONE else source_type


def _resolve_application(mode, application, effect_level, outdoor_time, sfx_level):
    """Fold the compact Application control into the level widget `mode` uses.

    An Application the mode does not define normalises to that mode's own level
    widget and warns, so a headless caller sees the mismatch instead of getting
    a caption it never asked for.
    """

    allowed = APPLICATION_BY_MODE.get(mode)
    if allowed is None:  # dry: nothing to apply
        return effect_level, outdoor_time, sfx_level

    if application not in allowed:
        if application != APPLICATION_NEUTRAL:
            logger.warning(
                "AKUSPACE: %s; using the mode's own level widget instead. "
                "Valid for %s mode: %s.",
                validate_choice(application, sorted(allowed), f"{mode} application"),
                mode,
                ", ".join(sorted(allowed)),
            )
        return effect_level, outdoor_time, sfx_level

    value = allowed[application]
    if mode == "room":
        return value, outdoor_time, sfx_level
    if mode == "outside":
        return effect_level, value, sfx_level
    return effect_level, outdoor_time, value


def _conditioned_prompt(
    text,
    space_mode,
    application,
    room_preset,
    effect_level,
    outdoor_time,
    sfx_level,
    source_type=SOURCE_NONE,
):
    mode = MODE_VALUES.get(space_mode, "room")
    effect_level, outdoor_time, sfx_level = _resolve_application(
        mode,
        application,
        effect_level,
        outdoor_time,
        sfx_level,
    )
    caption = build_caption(
        space_mode=mode,
        room_preset=room_preset,
        outdoor_time=outdoor_time,
        effect_level=effect_level,
        source_type=_normalise_source(source_type),
        sfx_level=sfx_level,
    )
    return compose_prompt(text, caption, enabled=mode != "dry")


class KoshiAKUSPACEPrompt:
    CATEGORY = "Koshi/Space"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _controls(),
            "optional": {
                # Optional so the six required widgets keep their positions in
                # graphs saved before this control existed; it lands last.
                **_source_control(),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "display_name": "Prompt",
                        "tooltip": "Optional visual prompt to extend with AKUSPACE conditioning.",
                    },
                ),
            },
        }

    def execute(
        self,
        space_mode,
        application,
        room_preset,
        effect_level,
        outdoor_time,
        sfx_level,
        source_type=SOURCE_NONE,
        prompt="",
    ):
        return (
            _conditioned_prompt(
                prompt,
                space_mode,
                application,
                room_preset,
                effect_level,
                outdoor_time,
                sfx_level,
                source_type,
            ),
        )


class KoshiAKUSPACETextEncode:
    CATEGORY = "Koshi/Space"
    FUNCTION = "execute"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("Conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIP output from the checkpoint, text-encoder, or LoRA loader."},
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Visual prompt. AKUSPACE appends the selected treatment before encoding.",
                    },
                ),
                **_controls(),
            },
            "optional": _source_control(),
        }

    def execute(
        self,
        clip,
        text,
        space_mode,
        application,
        room_preset,
        effect_level,
        outdoor_time,
        sfx_level,
        source_type=SOURCE_NONE,
    ):
        prompt = _conditioned_prompt(
            text,
            space_mode,
            application,
            room_preset,
            effect_level,
            outdoor_time,
            sfx_level,
            source_type,
        )
        return (encode_conditioning(clip, prompt),)


NODE_CLASS_MAPPINGS = {
    "Koshi_AKUSPACEPrompt": KoshiAKUSPACEPrompt,
    "Koshi_AKUSPACETextEncode": KoshiAKUSPACETextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_AKUSPACEPrompt": "◉ AKUSPACE Prompt",
    "Koshi_AKUSPACETextEncode": "◉ AKUSPACE Text Encode",
}
