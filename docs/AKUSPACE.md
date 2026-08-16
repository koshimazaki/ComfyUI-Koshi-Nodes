# AKUSPACE audio conditioning

AKUSPACE is an experimental spatial-audio LoRA control surface for LTX audio
generation. The reference audio supplies the source identity; the nodes append
only the selected acoustic treatment using the caption grammar learned during
training.

## Nodes

Both nodes appear under **Koshi → Space** and use the circular `◉` category
mark in the node title.

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

The Text field remains user-authored. The selected treatment is appended and
encoded when the workflow runs; the room overlay does not duplicate the prompt.

To inspect the generated suffix without covering the room visualization, connect
the modular node to Comfy's separate text viewer:

```text
AKUSPACE Prompt → Preview as Text
```

## Controls

- **Mode**: stepped Off / Room / Space / SFX fader.
- **Room**: Small, Club, Medium, Cathedral plus Low, Moderate, Heavy dry/wet.
- **Space**: Day or Night ambience.
- **SFX**: experimental Dual Delay at Low or High.

The holographic room is a relative visualization, not a physical room
measurement. Its controls snap to trained categories rather than implying
unsupported continuous conditioning. The Comfy overlay uses shorter faders and
tighter spacing than the standalone website controller so the room stays visible.

## Model and demo

- Interactive project page: [audiolora.dev](https://audiolora.dev/)
- Node source and installation: [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes)
- Interaction reference: [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle)

The LoRA checkpoint and LTX workflow are not bundled with the node pack. Add the
public Hugging Face model link here when the checkpoint is released.

## Status

AKUSPACE is pre-alpha 0.5. Validate the checkpoint and dry reference against the
trainer path before treating the Comfy graph as a production inference recipe.
