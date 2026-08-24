# AKUSPACE workflows — LTX-2.5 acoustic space control

Three graphs, API format. Drop into ComfyUI and queue, or POST to `/prompt`.

| workflow | what it does | nodes | time |
|---|---|---|---|
| `akuspace_a2a_treat_recording.json` | puts a space on **your own recording** | 24 | ~60 s |
| `akuspace_t2v_native_onepass.json` | generates picture **and** treated sound together | 38 | ~72 s |
| `akuspace_t2v_2stage_hq.json` | same, with a ×2 latent upscale + refine pass | 41 | ~60 s |

Measured on an RTX PRO 6000 (95 GB). Peak VRAM ~44 GB.

## Models

All from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (gated — accept
the licence first):

```
diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors   20.0 GB
text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors           14.3 GB
vae/ltx-2.5-audio-vae-bf16.safetensors                                               0.3 GB
vae/ltx-2.5-video-vae-bf16.safetensors                                               1.4 GB   (video graphs only)
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors        0.9 GB   (2-stage only)
```

Plus the adapter → `models/loras/akuspace-ltx25-v0.5.safetensors`.

The a2a graph is audio-only — it needs neither the video VAE nor the upscaler, so
~35 GB is enough for it.

## Requirements

- **ComfyUI ≥ 0.33.0.** `LTXVDualCFGGuider` is a core node in `comfy_extras/nodes_lt.py`;
  older builds (0.19.x ships on several cloud images) reject every graph here.
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
- This pack, at `ee5998d` or later — earlier commits do not contain
  `nodes/audio/aligned_ref.py`, and the import is guarded, so the pack loads
  cleanly with `AKUSPACEReferenceAudioAligned` silently absent.

## The one setting not to change

Both `LTXVDualCFGGuider` nodes run at `video_cfg=1.0, audio_cfg=1.0`.

The distilled transformer is CFG-distilled — guidance is baked into the weights.
Applying external CFG double-applies it: at 4.0 the audio rails completely
(RMS == peak == √2, every sample clipped). At `audio_cfg=7` the room effect drops
*below* the no-LoRA baseline. Leave both at 1.0.

## Why the 2-stage graph exists

Rendering 300+ frames at full resolution in one pass gives LTX-2.5 room to exercise
its native multishot behaviour — the clip cuts to a different framing mid-shot.
No CFG or prompt weighting reliably prevents this.

Rendering small first and upscaling fixes it structurally: composition is locked at
base resolution, and the 3-step refine (`0.85, 0.7250, 0.4219, 0.0`) can only sharpen
what is already there. It is also roughly 15× faster than the equivalent direct render.

## Levels

`space_mode` / `room_preset` drive the effect strongly. `application` and
`effect_level` currently do **not** behave as their names suggest — measured tail
energy runs `low ≥ moderate ≥ heavy` on every source tested. Use `Low` for the most
audible treatment until that is resolved.

`space_mode: Space` (outdoor) ignores `effect_level` entirely and is pinned to the
`outdoor_level` value in `nodes/audio/presets.json`.

## Inputs

`LoadImage` and `LoadAudio` resolve bare filenames against ComfyUI's own `input/`
directory. **Copy files there — do not symlink.** ComfyUI's path check rejects links
that escape `input/`, then drops the output nodes and still reports the job as
`success` with zero outputs in about 0.09 s.
