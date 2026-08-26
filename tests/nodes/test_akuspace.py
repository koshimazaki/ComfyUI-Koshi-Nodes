import logging

import pytest

from nodes.audio.akuspace import (
    APPLICATION_BY_MODE,
    APPLICATION_OPTIONS,
    MODE_OPTIONS,
    MODE_VALUES,
    NODE_DISPLAY_NAME_MAPPINGS,
    SOURCE_NONE,
    SOURCE_OPTIONS,
    KoshiAKUSPACEPrompt,
    KoshiAKUSPACETextEncode,
)
from nodes.audio.conditioning import SOURCE_VALUES


class FakeClip:
    def __init__(self):
        self.prompt = None

    def tokenize(self, prompt):
        self.prompt = prompt
        return {"tokens": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [["conditioning", {"tokens": tokens}]]


DEFAULTS = {
    "space_mode": "Room",
    "application": "Moderate",
    "room_preset": "medium_room",
    "effect_level": "mid",
    "outdoor_time": "day",
    "sfx_level": "low",
}


def test_nodes_use_space_category_and_circle_mark():
    assert KoshiAKUSPACEPrompt.CATEGORY == "Koshi/Space"
    assert KoshiAKUSPACETextEncode.CATEGORY == "Koshi/Space"
    assert NODE_DISPLAY_NAME_MAPPINGS == {
        "Koshi_AKUSPACEPrompt": "◉ AKUSPACE Prompt",
        "Koshi_AKUSPACETextEncode": "◉ AKUSPACE Text Encode",
    }


def test_prompt_variant_appends_effect_only_caption():
    output = KoshiAKUSPACEPrompt().execute(prompt="cinematic portrait", **DEFAULTS)[0]
    assert output == (
        "cinematic portrait, AKUSPACE in a medium reverberant room, "
        "moderate smooth reflections and a 1.9-second reverb decay, "
        "no background ambience"
    )


def test_text_encode_variant_uses_combined_prompt():
    clip = FakeClip()
    conditioning = KoshiAKUSPACETextEncode().execute(
        clip=clip,
        text="cinematic portrait",
        **DEFAULTS,
    )[0]
    assert clip.prompt.endswith("no background ambience")
    assert conditioning == [["conditioning", {"tokens": {"tokens": clip.prompt}}]]


def test_off_is_a_true_prompt_bypass():
    output = KoshiAKUSPACEPrompt().execute(
        prompt="cinematic portrait",
        **{**DEFAULTS, "space_mode": "Off", "application": "Off"},
    )[0]
    assert output == "cinematic portrait"


def _caption(**overrides):
    return KoshiAKUSPACEPrompt().execute(prompt="", **{**DEFAULTS, **overrides})[0]


@pytest.mark.parametrize(
    ("space_mode", "application", "fragment"),
    [
        ("Room", "Low", "gentle smooth reflections"),
        ("Room", "Moderate", "moderate smooth reflections"),
        ("Room", "Heavy", "heavy smooth reflections"),
        ("Space", "Day", "outdoors in daytime"),
        ("Space", "Night", "outdoors at night"),
        ("Sound effects", "Low", "gentle scattered grains"),
        ("Sound effects", "High", "heavy scattered grains"),
    ],
)
def test_each_mode_honours_its_own_applications(space_mode, application, fragment):
    assert fragment in _caption(space_mode=space_mode, application=application)


def test_application_map_covers_every_conditioning_mode():
    conditioning_modes = {MODE_VALUES[m] for m in MODE_OPTIONS} - {"dry"}
    assert set(APPLICATION_BY_MODE) == conditioning_modes
    for allowed in APPLICATION_BY_MODE.values():
        assert set(allowed) <= set(APPLICATION_OPTIONS)


@pytest.mark.parametrize(
    ("space_mode", "application"),
    [
        ("Room", "Day"),
        ("Room", "Night"),
        ("Room", "High"),
        ("Space", "Low"),
        ("Space", "Moderate"),
        ("Space", "Heavy"),
        ("Space", "High"),
        ("Sound effects", "Moderate"),
        ("Sound effects", "Heavy"),
        ("Sound effects", "Day"),
        ("Sound effects", "Night"),
    ],
)
def test_application_foreign_to_the_mode_warns_and_normalises(
    space_mode, application, caplog
):
    """Headless callers must be told, not silently handed another caption."""

    with caplog.at_level(logging.WARNING, logger="nodes.audio.akuspace"):
        output = _caption(space_mode=space_mode, application=application)

    assert "AKUSPACE" in caplog.text
    assert application in caplog.text
    # Normalises to the mode's own level widget, i.e. the DEFAULTS value.
    assert output == _caption(space_mode=space_mode, application="Off")


def test_neutral_application_is_silent_and_is_not_a_bypass(caplog):
    with caplog.at_level(logging.WARNING, logger="nodes.audio.akuspace"):
        output = _caption(space_mode="Room", application="Off")

    assert caplog.text == ""
    # Only space_mode="Off" bypasses; application="Off" still conditions.
    assert "medium reverberant room" in output


def test_no_mode_application_pair_raises():
    for space_mode in MODE_OPTIONS:
        for application in APPLICATION_OPTIONS:
            assert isinstance(_caption(space_mode=space_mode, application=application), str)


# ---------------------------------------------------------------------------
# source_type — the trained caption grammar starts with what the source IS
# ---------------------------------------------------------------------------


def test_source_type_defaults_to_effect_only_caption():
    """Graphs saved before this control existed must be byte-identical."""

    assert KoshiAKUSPACEPrompt().execute(prompt="a portrait", **DEFAULTS)[0] == (
        KoshiAKUSPACEPrompt().execute(prompt="a portrait", source_type="none", **DEFAULTS)[0]
    )


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        (
            "female spoken voice",
            "AKUSPACE female spoken voice in a medium reverberant room, "
            "moderate smooth reflections and a 1.9-second reverb decay, no background ambience",
        ),
        (
            "male spoken voice",
            "AKUSPACE male spoken voice in a medium reverberant room, "
            "moderate smooth reflections and a 1.9-second reverb decay, no background ambience",
        ),
    ],
)
def test_source_type_leads_the_trained_caption(source_type, expected):
    assert _caption(source_type=source_type) == expected


def test_source_options_are_the_public_subset():
    assert SOURCE_OPTIONS == [
        "female spoken voice",
        "male spoken voice",
    ]
    # Every displayed option comes from presets.json, never hand-typed.
    assert SOURCE_OPTIONS == SOURCE_VALUES


def test_source_type_survives_a_bypass_and_a_dry_caption():
    off = KoshiAKUSPACEPrompt().execute(
        prompt="a portrait",
        source_type="female spoken voice",
        **{**DEFAULTS, "space_mode": "Off"},
    )[0]
    assert off == "a portrait"


def test_source_type_is_optional_on_both_nodes():
    for cls in (KoshiAKUSPACEPrompt, KoshiAKUSPACETextEncode):
        spec = cls.INPUT_TYPES()
        assert "source_type" not in spec["required"], f"{cls.__name__}: must not shift widget order"
        assert "source_type" in spec["optional"], f"{cls.__name__}: source_type missing"


def test_text_encode_passes_source_type_through_to_clip():
    clip = FakeClip()
    KoshiAKUSPACETextEncode().execute(
        clip=clip, text="a portrait", source_type="male spoken voice", **DEFAULTS
    )
    assert clip.prompt.startswith("a portrait, AKUSPACE male spoken voice in a medium")
