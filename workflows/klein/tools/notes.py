"""MarkdownNote text placed on the UI canvas (title, pos, size, text)."""

AGENT_SLOTS = """## Agent slots (Genobi fills these in the API JSON)

**Text widgets**

| slot | node id | input | default |
|---|---|---|---|
| base prompt | `4` CLIPTextEncode | `text` | cafe scene with a glass of iced coffee |
| element name | `21` CLIPTextEncode (SAM 3.1) | `text` | `drink` |
| instruction | `40` PrimitiveStringMultiline | `value` | `hold a banana instead of the drink` |

Seeds: `7.seed` (base), `51.seed` (edit).

**Optional images** (bypassed here, absent from the API JSON until Genobi adds them)

| slot | node id | wiring when used |
|---|---|---|
| external base image | `10` LoadImage | `11.on_true = ["10", 0]` and `11.switch = true` |
| reference image | `49` LoadImage | `50["images.image1"] = ["49", 0]`; cite it as `<ref_image_1>` in the instruction |

The element name accepts SAM 3.1 syntax: `drink:2` keeps up to two detections.
"""

HOW_IT_WORKS = """## How the three stages connect

1. **Base** — FLUX 3 Image Klein distilled, 1024 x 1024, 16 steps, cfg 1, euler / simple (the bundled template settings). The switch `11` lets stages 2 and 3 run on a loaded image instead; Klein then never runs.
2. **Find** — SAM 3.1 (`SAM3_Detect`, core) segments the element named in `21`. The mask is grown 24 px, then blurred for a soft edge (`FeatherMask` only feathers the canvas border, so it is not used). The box goes out as `BOUNDING_BOX` and as a STRING via `PreviewAny`.
3. **Swap** — the SAM box becomes a FLUX 3 layout element (`CreateBoundingBoxes` → `Flux3LayoutPrompt`), so the instruction is tied to `<obj_1>`. The base is `<ref_image_0>` in `ReferenceMultiLatent`, as in the official edit template. Klein regenerates the whole frame, so the result is composited back over the base through the feathered mask.

**Open questions (need a GPU run)**: does Klein keep pixels outside `<obj_1>` well enough to skip the composite; is 24 px of growth enough for an object larger than the drink; does an empty SAM result (nothing found) need a guard. Details in `workflows/klein/README.md`.
"""

NOTES = [
    ("Agent slots", [0, 1320], [900, 520], AGENT_SLOTS),
    ("How it works · open questions", [940, 1320], [860, 520], HOW_IT_WORKS),
]
