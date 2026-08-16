"""AKUSPACE prompt and text-encoding controls for the spatial-audio LoRA."""

from .conditioning import (
    EFFECT_LEVELS,
    OUTDOOR_TIMES,
    ROOM_PRESETS,
    SFX_LEVELS,
    build_scene,
    compose_prompt,
    encode_conditioning,
)


MODE_OPTIONS = ["Off", "Room", "Space", "Sound effects"]
APPLICATION_OPTIONS = ["Off", "Low", "Moderate", "Heavy", "Day", "Night", "High"]
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


def _resolve_application(mode, application, effect_level, outdoor_time, sfx_level):
    if mode == "room":
        effect_level = ROOM_APPLICATION.get(application, effect_level)
    elif mode == "outside":
        outdoor_time = SPACE_APPLICATION.get(application, outdoor_time)
    elif mode == "sfx":
        sfx_level = SFX_APPLICATION.get(application, sfx_level)
    return effect_level, outdoor_time, sfx_level


def _conditioned_prompt(
    text,
    space_mode,
    application,
    room_preset,
    effect_level,
    outdoor_time,
    sfx_level,
):
    mode = MODE_VALUES.get(space_mode, "room")
    effect_level, outdoor_time, sfx_level = _resolve_application(
        mode,
        application,
        effect_level,
        outdoor_time,
        sfx_level,
    )
    scene = build_scene(
        space_mode=mode,
        room_preset=room_preset,
        outdoor_time=outdoor_time,
        effect_level=effect_level,
        source_type="",
        sfx_level=sfx_level,
    )
    return compose_prompt(text, scene["caption"], enabled=mode != "dry")


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
    ):
        prompt = _conditioned_prompt(
            text,
            space_mode,
            application,
            room_preset,
            effect_level,
            outdoor_time,
            sfx_level,
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
