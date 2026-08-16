from nodes.audio.akuspace import (
    NODE_DISPLAY_NAME_MAPPINGS,
    KoshiAKUSPACEPrompt,
    KoshiAKUSPACETextEncode,
)


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
