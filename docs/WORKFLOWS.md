# Example workflows and pipelines

## Shipped workflow files

The general examples in [`workflows/`](../workflows/) cover motion-driven V2V,
temporal coherence, OLED export, and sprite sheets:

- `koshi_v2v_ultimate.json` — full V2V with motion, temporal control, and colour match.
- `koshi_v2v_complete.json` — complete V2V pipeline.
- `koshi_v2v_motion.json` — motion-focused V2V.
- `koshi_v2v_temporal.json` — temporal-coherence V2V.
- `koshi_v2v_pure.json` — minimal V2V setup.
- `koshi_oled_sprite_pipeline.json` — image → dither → OLED preview → export.
- `koshi_sprite_sheet.json` — video → sprite sheet.

The three public LTX-2.5 graphs live in
[`workflows/akuspace/`](../workflows/akuspace/). Their README lists required
models, inputs, measured hardware, and the commands for verification, batch
execution, and an adapter-on/off A/B.

## Quick pipelines

Motion animation:

```text
▄▀▄ Schedule → ▄▀▄ Motion Engine → KSampler
                                      ↓
                              ▄▀▄ Feedback (loop)
```

Audio-driven motion:

```text
LoadAudio → ▄▀▄ Audio → Motion → ▄▀▄ Motion Engine → KSampler
```

Stacked effects and export:

```text
Image → ░▀░ Koshi Effects → ░▀░ KN Bloom → ░▒░ KN SIDKIT OLED
```

AKUSPACE with a reusable prompt string:

```text
Base Prompt → ◉ AKUSPACE Prompt → CLIP Text Encode → Conditioning
```

AKUSPACE with integrated encoding:

```text
LoRA-patched CLIP + Text → ◉ AKUSPACE Text Encode → Conditioning
```

One-pass reference audio:

```text
UNETLoader → Load LoRA → ◉ AKUSPACE Reference Audio (aligned) → LTXV Dual CFG Guider
                         ↑ dry recording
LTXV Empty Latent Audio → Concat A/V Latent
```

The audio target must stay empty in this last graph. A zero noise mask pins the
audio and leaves the reference-conditioned adapter nothing to transform.
