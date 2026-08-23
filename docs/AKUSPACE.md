# AKUSPACE audio conditioning

AKUSPACE is an experimental spatial-audio LoRA control surface for LTX audio
generation. The reference audio supplies the source identity; the nodes append
only the selected acoustic treatment using the caption grammar learned during
training.

## Nodes

All three appear under **Koshi → Space** and use the circular `◉` category
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

### ◉ AKUSPACE Reference Audio (aligned)

Node id `AKUSPACEReferenceAudioAligned` — deliberately unprefixed, so a graph
built against the standalone copy in the AKUSPACE session kit resolves against
the pack and vice versa.

Feeds a **dry recording** into the generation as a time-aligned in-context
reference, which is what makes the a2a IC-LoRA run inside a video render rather
than as a separate pass:

```text
UNETLoader → Load LoRA (AKUSPACE) → ◉ AKUSPACE Reference Audio (aligned) → Guider
                                     ↑ dry audio, trimmed to the clip length
```

**Why it is not core's `LTXVReferenceAudio`.** That node is an ID-LoRA
(speaker-identity) node: it prepends the reference with **negative** temporal
positions, ending just before t=0. The LTX trainer's a2a IC-LoRA does the
opposite, identically in training (`flexible.py::_apply_reference_condition` →
`_get_audio_positions`) and in inference (`validation_runner` →
`get_patch_grid_bounds`): the reference gets **the same positions a target of
that length would get, from t=0** — reference token *i* at the time of target
token *i*. That time-aligned layout is what AKUSPACE learned, and it is why it
holds timing to 16–24 ms. Through the stock node every reference token sits
`T_ref+1` latents from where the adapter learned to look.

Mechanically it is a narrow monkeypatch of `LTXAVModel._process_input`, active
only when the conditioning carries `akuspace_aligned: True`. Stock conditioning
is untouched, so both nodes work side by side — which is the point: they make a
clean A/B, one node apart. If a future ComfyUI core changes that method's return
shape the patch raises rather than silently rendering the stock convention.

`reference_guidance_scale` defaults to **0** because trainer inference had no
such term. Above 0 it costs an extra forward pass per step.

**The audio latent must be empty.** Pair this node with `LTXVEmptyLatentAudio`,
never with a pinned/masked audio latent: a mask that holds the audio fixed stops
the adapter transforming anything.

## Controls

- **Mode**: stepped Off / Room / Space / SFX fader.
- **Room**: Small, Club, Medium, Cathedral plus Low, Moderate, Heavy dry/wet.
- **Space**: Day or Night ambience.
- **SFX**: experimental Dual Delay at Low or High.
- **Source** *(optional)*: what the dry recording is. The trained captions
  begin with it — `AKUSPACE female spoken voice in a small bathroom-like
  room, …` — so setting it puts the caption fully in distribution. It defaults
  to `none`, which keeps the effect-only caption and leaves graphs saved before
  this control existed byte-identical. Options come from `presets.json`, so they
  cannot drift from the trained vocabulary: male/female spoken voice, electronic
  rhythm loop, handclaps, solo piano phrase, piano and saxophone phrase, citola
  string phrase, and granular sound effect (marked experimental).

**Space mode always reads *gentle*.** Outdoor cells were only trained gentle and
heavy, and an ambience bed is a recording rather than a reverb tail, so it
scales down but not up. `resolve_level_key` forces the level rather than letting
an untrained caption through.

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

## Workflows

Ten runnable LTX-2.5 graphs ship in [`workflows/akuspace/`](../workflows/akuspace/).
Drag any of them into ComfyUI.

| file | shape | audio | adapter |
|---|---|---|---|
| `akuspace_ltx25_i_a2v_aligned` | image + audio → video | dry reference, **empty** target | AKUSPACE, aligned node |
| `akuspace_ltx25_flf_a2v_aligned` | first + last frame + audio → video | dry reference, **empty** target | AKUSPACE, aligned node |
| `akuspace_ltx25_flf_a2v_stock` | ↑ identical | ↑ identical | AKUSPACE, core `LTXVReferenceAudio` |
| `akuspace_ltx25_flf_a2v_nolora` | ↑ identical | ↑ identical | none |
| `akuspace_ltx25_flf_twopass` | first + last frame + audio | already roomed, **pinned** | none |
| `akuspace_ltx25_flf_split_chain` | first + last frame, **separate audio pass** | dry reference, empty target, audio-only model | AKUSPACE, aligned node |
| `akuspace_ltx25_t2v_split_chain` | **text to video** — no image input at all | dry from the render, empty target, audio-only pass | AKUSPACE, aligned node |
| `akuspace_ltx25_ia2v_roomed_prompt` | image + audio → video, **roomed first** | AKUSPACE rooms it, then it is pinned | AKUSPACE, audio-only pass |
| `akuspace_ltx25_ia2v_roomed_encode` | ↑ identical, one-node conditioning | ↑ identical | ↑ identical |
| `video_ltx25_AKUSPACE_flf_ia2v` | first + last frame + your audio (subgraph style) | supplied, pinned | none |

The `ia2v_roomed_*` pair answers "can the image+audio→video path go through the
LoRA?" — not directly, because a pinned audio latent is never denoised, so the
adapter would have nothing to act on. Instead the adapter runs **first** in its
own audio-only pass and the video is pinned to its **output**, so the picture
lip-syncs to the audio you actually ship. The prompt is written from the anchor
image by `TextGenerateLTX2Prompt`, with the AKUSPACE caption appended *after* the
enhancer (it paraphrases the trigger and the level words). The two variants
differ only in which conditioning node is used: `_prompt` uses
`◉ AKUSPACE Prompt` → a stock `CLIP Text Encode`, `_encode` uses
`◉ AKUSPACE Text Encode` → CONDITIONING in one node.

`akuspace_ltx25_flf_split_chain` is the one that uses **both** AKUSPACE nodes.
The video and audio are generated as two separate passes in one graph, so the
audio chain reads the trained caption **alone** — no shot grammar, nothing the
adapter did not see in training — while the treatment still appears in the video
prompt so the picture agrees about the space. `◉ AKUSPACE Prompt` holds the
selection (no Prompt input → the caption alone) and `◉ AKUSPACE Text Encode` at
**Mode = Off** encodes it verbatim for the audio chain. The audio pass runs
through `LTXVAudioOnlyModel`, which severs the audio↔video cross-attention the
LoRA does not patch — so the treatment should land harder there than in a joint
render. One queue produces the BEFORE video, the AFTER video and both audio
files.

Rows 2–4 are the same graph one node apart — the A/B described above, plus a
LoRA-off control. Render all three and correlate each output against the dry
input: r 0.5–0.75 at ≈ −20 ms means the adapter followed the reference **and**
transformed it; r ≈ 0 means the reference was ignored; r ≈ 1 means the dry was
copied. That settles the question by measurement rather than by ear.

They are generated and checked by
[`build_akuspace_workflows.py`](https://github.com/koshimazaki/AUDIO-LTX-LORA)
in the AKUSPACE repo, which validates every class, socket and combo value
against a live `object_info` dump before writing anything.

## Model and demo

- Model weights: [KoshiMazaki/akuspace-ltx25](https://huggingface.co/KoshiMazaki/akuspace-ltx25)
  → `models/loras/akuspace-ltx25-v0.5.safetensors`
- Interactive project page: [akuspace.pages.dev](https://akuspace.pages.dev)
- Node source and installation: [ComfyUI-Koshi-Nodes](https://github.com/koshimazaki/ComfyUI-Koshi-Nodes)
- Interaction reference: [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle)

The LoRA checkpoint is not bundled with the node pack.

## Status

AKUSPACE is pre-alpha 0.5. Validate the checkpoint and dry reference against the
trainer path before treating the Comfy graph as a production inference recipe.
