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

### Mode and Application

Application is a single control shared by every mode, so each mode reads only
the values it defines:

| Mode | Application values | Falls back to |
|------|--------------------|---------------|
| Off | *(none — conditioning is bypassed)* | — |
| Room | Low, Moderate, Heavy | Dry / wet widget |
| Space | Day, Night | Space widget |
| SFX | Low, High | Dry / wet widget |

`Application = Off` is **not** a bypass — it means "no override, use the mode's
own level widget". Only `Mode = Off` bypasses conditioning and returns the
prompt untouched.

The graph UI keeps Mode and Application in step. API and headless callers can
set a pair a mode does not define (say Room + Night); those normalise to the
mode's own level widget and log a warning rather than silently returning a
different caption.

### Widget loading

The room visualization ships as a prebuilt bundle (`js/akuspace-widget.mjs`,
~980KB with a Vue runtime and Three.js). `js/akuspace-loader.js` imports it on
demand the first time an AKUSPACE node appears, so graphs without these nodes
pay nothing at startup — the same approach ComfyUI core uses for its Three.js
Load3D extension. The `.mjs` extension is deliberate: it keeps the bundle out
of ComfyUI's `**/*.js` extension auto-load glob.

Third-party licences for the bundled libraries are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Model and demo

- Interactive project page: [audiolora.dev](https://audiolora.dev/)
- Node source and installation: [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes)
- Interaction reference: [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle)

The LoRA checkpoint and LTX workflow are not bundled with the node pack. Add the
public Hugging Face model link here when the checkpoint is released.

## Status

AKUSPACE is pre-alpha 0.5. Validate the checkpoint and dry reference against the
trainer path before treating the Comfy graph as a production inference recipe.
