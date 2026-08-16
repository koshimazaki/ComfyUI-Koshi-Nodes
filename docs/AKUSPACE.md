# AKUSPACE audio conditioning

AKUSPACE is an experimental spatial-audio LoRA control surface for LTX audio
generation. The reference audio supplies the source identity; the nodes append
only the selected acoustic treatment using the caption grammar learned during
training.

## Nodes

### ◉ AKUSPACE Prompt

Use this modular variant beside camera, lighting, or other prompt-building
nodes:

```text
Base prompt → AKUSPACE Prompt → CLIP Text Encode → Conditioning
```

With no Base prompt connection, it emits only the AKUSPACE treatment. Off is a
true prompt bypass.

### ◉ AKUSPACE Text Encode

Use this convenience variant in place of the native CLIP Text Encode node:

```text
LoRA-patched CLIP ─┐
                   ├→ AKUSPACE Text Encode → Conditioning
Editable Text ─────┘
```

The Text field remains user-authored. The controller shows a live read-only
preview of the combined prompt; the selected treatment is appended and encoded
when the workflow runs.

## Controls

- **Mode**: stepped Off / Room / Space / SFX fader.
- **Room**: Small, Club, Medium, Cathedral plus Low, Moderate, Heavy dry/wet.
- **Space**: Day or Night ambience.
- **SFX**: experimental Dual Delay at Low or High.

The holographic room is a relative visualization, not a physical room
measurement. Its controls snap to trained categories rather than implying
unsupported continuous conditioning.

## Model and demo

- Interactive project page: [audiolora.dev](https://audiolora.dev/)
- Node source and installation: [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes)
- Interaction reference: [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle)

The LoRA checkpoint and LTX workflow are not bundled with the node pack. Add the
public Hugging Face model link here when the checkpoint is released.

## Status

AKUSPACE is pre-alpha 0.5. Validate the checkpoint and dry reference against the
trainer path before treating the Comfy graph as a production inference recipe.
