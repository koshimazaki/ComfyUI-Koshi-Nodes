# AKUSPACE workflows — LTX-2.5 acoustic space control

Three verified graphs in ComfyUI API format, plus one draft. Queue them with the
public runner scripts or POST them directly to `/prompt`.

| workflow | what it does | nodes | time |
|---|---|---|---|
| `akuspace_a2a_treat_recording.json` | puts a space on **your own recording** | 24 | ~60 s |
| `akuspace_t2v_native_onepass.json` | generates picture **and** treated sound together | 38 | ~72 s |
| `akuspace_t2v_2stage_hq.json` | same, with a ×2 latent upscale + refine pass | 41 | ~60 s |
| `akuspace_rooms_chain.json` (UI) + `api/akuspace_rooms_chain_api.json` | **draft, not yet run:** one character through three rooms, three chained segments, one preset per room | 137 UI / 90 API | unknown |

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

`LoadAudio` resolves bare filenames against ComfyUI's own `input/` directory.
**Copy files there — do not symlink.** ComfyUI's path check rejects links that
escape `input/`, then drops the output nodes and can still report the job as
`success` with zero outputs.

The default graphs reference:

- `submission_dry_voice.wav`
- `dj_dry_voice.wav`
- `dry-base-generated-voice.mp3`

Override those node inputs with `--set NODE_ID.audio=filename.wav`, or edit a
copy of the graph.

## Verify and run

From the repository root:

```bash
python3 scripts/akuspace/verify_workflows.py
python3 scripts/akuspace/run_batch.py --dry-run
python3 scripts/akuspace/run_ab.py --dry-run
```

When ComfyUI is running on the LTX setup script's default port:

```bash
export COMFY_URL=http://127.0.0.1:8189
python3 scripts/akuspace/run_batch.py
python3 scripts/akuspace/run_ab.py
```

The A/B uses the native one-pass graph by default. It changes only the
`LoraLoaderModelOnly.strength_model` value (`1` versus `0`) and output prefixes;
the seed, prompt, reference, sampler, and every other graph input remain equal.

## Rooms chain (draft, not yet run on a GPU)

One character walks through three rooms. LTX-2.5 renders three chained segments,
each with its own AKUSPACE room preset and its own dry reference voice, so the
voice sounds like the room it is in. The graph is generated, not hand-edited:

```bash
python3 scripts/akuspace/build_rooms_chain.py          # writes the UI and API files
python3 scripts/akuspace/build_rooms_chain.py --check  # fails if either file is stale
curl -s http://127.0.0.1:8189/object_info > /tmp/object_info.json
python3 scripts/akuspace/check_rooms_chain.py /tmp/object_info.json   # never queues
```

`check_rooms_chain.py` checks the node schemas, every input, link type, combo value
and number range of the API graph, and of four variants with the bypassed groups
switched on. It does not render anything. Run it again on the GPU box, where
ComfyUI-LTXVideo is installed, before the first queue.

### How the segments join

| segment | room preset (default) | frames | new frames kept | ends at |
|---|---|---|---|---|
| 1 | `cathedral` | 241 (8·30+1) | 0–240 | 10.04 s |
| 2 | `small_room` | 257 (8·32+1) | 17–256 | 20.04 s |
| 3 | `empty_club` | 257 (8·32+1) | 17–256 | 30.04 s |

Total 721 frames at 24 fps. 240 frames (10.0 s) is not 8k+1, so segment 1 uses 241;
segments 2 and 3 are 17 overlap frames plus exactly 240 new ones.

- **Continuation.** `ImageFromBatch` takes the last 17 decoded frames of the
  previous segment. Core `LTXVAddGuide` pins them at `frame_idx 0`, strength 1.0,
  on the new segment's empty video latent, before `LTXVConcatAVLatent`. After
  sampling, `LTXVSeparateAVLatent` → `LTXVCropGuides` strips the guide tokens.
- **Join.** Segments 2 and 3 drop their first 17 frames (`ImageFromBatch 17, 240`)
  and the matching 0.7083 s of audio (`TrimAudioDuration`). `ImageBatch` ×2 and
  `AudioConcat` ×2 make one picture track and one audio track. `CreateVideo` →
  `SaveVideo` writes `video/AKU_rooms_chain`, plus a FLAC of the joined audio.
  Each segment is also saved whole (`video/AKU_rooms_segN_<room>`, overlap
  included) for checking the seams or cutting in an editor.
- **Shared once.** UNET, AKUSPACE LoRA, Gemma CLIP, both VAEs, the negative prompt,
  the stage-1 sampler and sigmas. Per segment: visual prompt →
  `Koshi_AKUSPACEPrompt` → `CLIPTextEncode` → `LTXVConditioning` →
  `AKUSPACEReferenceAudioAligned` (own `LoadAudio` + `TrimAudioDuration`, sized
  to the segment) → `LTXVDualCFGGuider` at 1/1 → 8-step `SamplerCustomAdvanced`.
  Seeds are the one-pass seed, +1 and +2.
- **Reference voice.** Every segment defaults to `dj_dry_voice.wav` from 0 s, so
  each room repeats the same line. For one continuous monologue, use a dry
  recording of at least 30.1 s and set `start_index` to 9.3333 on node 207 and to
  19.3333 on node 307. Segment *n* frame 0 sits at global frame 224·(n−1).
- **832×464.** As in the verified graphs, `EmptyLTXVLatentVideo` floors the
  height to a multiple of 32, so the latent is 26×14 and frames decode at 832×448.

### Optional groups (bypassed by default)

Toggle a group with its right-click menu (*Bypass group nodes*). The API file is
the default state: all three groups off.

- **Klein first frame** (per segment): `LoadImage` → `LTXVPreprocess 18`. On
  segment 1 it feeds `LTXVImgToVideoInplace` at 0.7, as in the official I2V
  template. Segments 2 and 3 have no free first frame, since frame 0 is the
  hand-off. Their Klein still goes to an `LTXVAddGuide` at frame 17 (the first
  kept frame), strength 0.7. Generate it from the previous segment's last frame,
  or the cut will jump. Move `frame_idx` later to use it as a room-reveal keyframe.
- **Depth/canny guide** (per segment, plus the shared loader node 6): World Builder
  · Capture → `ImageFromBatch` → `LTXAddVideoICLoRAGuide`. The loader is
  `LTXICLoRALoaderModelOnly` with `ltx-2.3-22b-ic-lora-union-control-ref0.5`, and
  its `latent_downscale_factor` output is linked into each guide. **Enable
  node 6 whenever any depth guide is on**: a bypassed loader leaves that input
  empty. `depth` is wired; re-link the Capture's `canny` output to swap. Segment
  1 uses all 241 capture frames at frame 0. Segments 2 and 3 use the first 233
  at frame 17, because a 241-frame guide there would run past the 257-frame latent
  (Capture allows at most 241 frames). World Builder · Acoustics sits beside each
  Capture and previews its suggested preset. To let it drive the room, link its
  `space_mode` / `application` / `room_preset` outputs onto that segment's
  `Koshi_AKUSPACEPrompt` widgets.
- **Stage-2 ×2 upscale + refine** (one group across all segments): latent
  upsampler, re-pair with the stage-1 audio latent, 3 steps `0.85 → 0`. Enable
  every node in it or none. When it is on, the next segment's tail comes from the
  upscaled decode, and audio still comes from stage 1: bypassing stage 2 would
  otherwise feed a video latent into the audio decoder. The verified one-pass
  graph decodes audio from stage 2 instead.

### Research behind the draft (ComfyUI-LTXVideo `220e1d4`, core v0.42.5)

1. **Continuing a generation.** The pack has `LTXVExtendSampler`, with
   `frame_overlap` min 16, step 8, and overlap strength 0.5 by default. It also
   has `LTXVLoopingSampler`. Both are video-only. The looping sampler raises on
   audio-video latents. The extend sampler unpacks a 5-D video latent, builds the
   new chunk with `EmptyLTXVLatentVideo`, so no audio stream is produced, and
   cross-fades with `LinearOverlapLatentTransition`. For a joint audio-video
   continuation, the supported path is core `LTXVAddGuide` + `LTXVCropGuides`,
   the pair used by the official `video_ltx2_5_flf2v` template.
   `LTXVAddGuide` takes 8n+1 frames and crops anything else down. Multi-frame
   guides away from frame 0 snap to 8k+1, and the node refuses a combined AV
   latent, so it goes before `LTXVConcatAVLatent`. The overlap is 17 frames
   (8·2+1, three latent frames). **Audio restarts every segment.** No audio
   continuation node exists for this. Each segment's sound comes from its own
   reference voice and room preset, and the previous room's reverb tail is cut
   at the seam. Core v0.42.5 also has `LTXVAddLatentGuide` (latent-space, no
   decode/encode). It is not used here: mid-clip latents carry the causal
   first-frame mismatch that `LTXVExtendSampler` works around by dropping a
   latent, and the deployment pins core v0.41.0, where its presence was not
   checked.
   One audio carry-over is possible but not wired. The LTX-2.3 `video_ltx2_3_ia2v`
   template holds an audio track fixed with `TrimAudioDuration` →
   `LTXVAudioVAEEncode` → `SetLatentNoiseMask` (`SolidMask` 0) →
   `LTXVConcatAVLatent`. Applied only to the first 0.7083 s, it could pin the
   previous segment's last audio as context. That needs a partial-length mask
   joined to an empty audio latent, which no core node provides directly. Those
   frames are trimmed anyway, and pinning the whole audio target would stop the
   LoRA from treating it (`verify_workflows.py` rejects `SetLatentNoiseMask` for
   that reason).
2. **Depth/canny for 2.5.** No LTX-2.5-native union-control IC-LoRA exists at
   this ref. The pack's own `2.5/LTX-2.5_ICLoRA_Union_Control_Distilled.json`
   loads the **LTX-2.3** `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`
   onto the 2.5 distilled transformer, via `LTXICLoRALoaderModelOnly` +
   `LTXAddVideoICLoRAGuide`. The only 2.5-native IC-LoRAs it ships are SDR-to-HDR,
   Day-To-Night and Refine-Details. The core templates `video_ltx2_depth_to_video`
   and `video_ltx2_canny_to_video` are LTX-2, not 2.5. The LTX-2.3 control
   template is `video_ltx2_3_ic_lora` (`GetICLoRAParameters`, `LTXVAddGuide`).
   A gated `Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render` (RGB playblast in) is
   reported on Hugging Face. Nothing in the pack at `220e1d4` mentions it, and it
   was not downloaded or checked here. camera-director's `sources.lock` drives it
   with the same `LTXICLoRALoaderModelOnly` + `LTXAddVideoICLoRAGuide` pair, so the
   depth slot can take it by swapping the LoRA file on node 6 and linking the
   Capture's `frames` output instead of `depth`. Its downscale factor and frame
   rules are unverified.
3. **Official 2.5 templates.** `comfyui-workflow-templates` 0.11.96 has no
   "all-in-one", extend or keyframe template for LTX-2.5. It ships
   `video_ltx2_5_t2v`, `_i2v` and `_flf2v`, plus `api_ltx2_5_*` for the paid API
   nodes. Each is one subgraph with groups named Model, Prompt, Prompt
   Enhancement, Video Settings, Empty Latent, Generate Low Resolution, Latent
   Upscale and Generate High Resolution. They use the samplers this pack already
   uses: `euler_ancestral`, the same `ManualSigmas`, `LTXVDualCFGGuider` 1/1,
   `VAEDecodeTiled`, `LTXVAudioVAEDecode`, `CreateVideo`. I2V feeds
   `LTXVPreprocess 18` into `LTXVImgToVideoInplace` (0.7 at stage 1, 1.0 at
   stage 2). FLF2V uses `LTXVAddGuide` at 0 and −1, strength 0.7, then
   `LTXVCropGuides`. The draft follows these conventions but stays flat, without
   a subgraph, so the per-segment wiring stays visible.

### Validation done without a GPU

- Core and Koshi nodes (every node in the API graph): checked against
  `/object_info` of a CPU ComfyUI v0.42.5 with this pack loaded.
- `LTXICLoRALoaderModelOnly`, `LTXAddVideoICLoRAGuide`: checked against
  ComfyUI-LTXVideo source at `220e1d4` (`iclora.py`).
- `DirectorSceneCapture`, `DirectorSceneAcoustics`: checked against camera-director
  `feat/world-builder-depth-acoustics` source.
- `verify_workflows.py`, `run_batch.py --dry-run` and the pack's pytest suite pass.

### Unverified until a GPU run

- **Seam quality.** Whether a 17-frame guide at strength 1.0 with the 8-step
  distilled schedule gives a cut-free hand-off at 10.04 s and 20.04 s. If the
  overlap freezes or flickers, try strength 0.7, then 0.5 (the extend sampler's
  default), or 25 overlap frames (change `OVERLAP` in the builder).
- **Guides and AKUSPACE together.** The aligned-reference patch of
  `LTXAVModel._process_input` has never run with `keyframe_idxs` from
  `LTXVAddGuide`.
- **Audio.** Whether decoded audio is exactly frames/24 long. Any shortfall
  accumulates as A/V drift across the two `AudioConcat` joins. Also untested: how
  the hard room change at each seam sounds.
- **Identity and story.** The character can drift, since only 17 frames of
  context carry over. Whether the prompts' door-to-door timing lands inside each
  10 s is untested.
- **Optional groups.** None has run: the Klein keyframe at frame 17, the 2.3
  union IC-LoRA stacked with the AKUSPACE LoRA on 2.5 (ref0.5 on a 26×14 latent
  gives 13×7), the 233-frame mid-clip depth guide, and stage 2 with stage-1 audio.
- **Core version.** The deployment pins core v0.41.0 (`98cc0bf`). Nodes here
  that the verified graphs do not use (`LTXVAddGuide`, `LTXVCropGuides`,
  `LTXVImgToVideoInplace`, `LTXVPreprocess`, `ImageFromBatch`, `ImageBatch`,
  `AudioConcat`, `LoadImage`) were checked on v0.42.5 only.
- **Frontend.** The UI file was generated, not saved from the canvas. It has not
  been opened in the frontend, and its canvas export has not been compared with
  the API file.
- **Memory and time.** VRAM per segment at these settings is unknown: this draft
  has not been run, and the ~44 GB measured for the one-pass graph does not
  transfer to it. Host RAM for the joined clip is about 3.2 GB of float32 frames
  (721 × 832 × 448 × 3 × 4 bytes), and about 12.9 GB with stage 2 on, before the
  per-segment copies. Render time is unknown.
