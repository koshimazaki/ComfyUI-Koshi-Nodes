# Koshi Audio Nodes

Audio/video → motion for the Koshi pipeline.

## ▄▀▄ KN Audio → Motion (`Koshi_AudioMotionSchedule`)

Turns audio (or video) features into a **`KOSHI_MOTION_SCHEDULE`** that plugs
straight into **KN Motion Engine** (`Koshi_MotionEngine`), plus a Deforum-style
zoom string for use elsewhere.

This node is a **bridge, not another analyzer.** Community packs (Fill-Nodes,
Yvann, RyanOnTheInside) already extract beats / envelopes / band energy — but they
modulate *decoded images* as a post-effect. None of them drive the *latent* motion
engine. This node closes that gap: it reuses existing analysis and maps it onto
Koshi's latent-space motion.

### Inputs / Outputs

| | |
|---|---|
| **Output** `motion_schedule` | `KOSHI_MOTION_SCHEDULE` → wire to `Koshi_MotionEngine.motion_schedule` |
| **Output** `zoom_schedule_string` | `STRING`, e.g. `0:(1.0), 5:(1.16), ...` (Deforum-compatible) |

### Three sources (the `source` widget)

1. **`analysis_json`** — paste an analysis JSON. **Comfy Cloud safe** (numpy only).
   Auto-detects three formats:
   - **BFL dashboard** `AudioAnalysisResult` (`markers[]` + dense `waveform[]`).
   - **Fill-Nodes** `FL_Audio_Reactive_Envelope` (`{"envelope":[...], "total_frames":N}`).
   - **Fill-Nodes** beat/drum times (`beat_times` / `kick_times` / `snare_times` / `hihat_times`).
2. **`audio`** — a ComfyUI `AUDIO` input, analyzed in-node with a numpy STFT
   band-split (low `<180Hz`, mid, high `>3600Hz`). No librosa required.
3. **`video`** — an MP4 path (`video_path`): per-frame brightness + motion
   magnitude + detail via OpenCV. This is the *"extract motion from an existing
   clip"* mode. Requires a working `opencv-python`.

### Feature → motion mapping

Band energies are `0..1`, scaled by each gain (set a gain to `0` to disable, or
negative to invert):

| Band | Drives | Default gain |
|------|--------|--------------|
| `low` (bass) | `zoom` (push-in pulse) | `0.12` |
| `high` | `angle` (rotation) | `0.0` |
| `mid` | `translation_x` (drift) | `0.0` |
| `amplitude` | `strength` (denoise) | `0.0` |

`feature` chooses the driver: `auto` (waveform if present, else markers),
`waveform` (dense continuous envelope — punchy, default for BFL/Fill envelope),
or `markers` (sparse impacts interpolated as keyframes — for beat/drum-times).
`smoothing` (0–1) tames jitter; values are clamped to the engine's ranges.

### Wiring (Cloud / signature tier)

```
[BFL dashboard analysis JSON]
        │ (paste into analysis_json)
        ▼
 KN Audio → Motion ──KOSHI_MOTION_SCHEDULE──▶ KN Motion Engine ──LATENT──▶ sampler/VAE
        └────────────zoom_schedule_string────▶ (optional: external Deforum tools)
```

Set the engine's `frame_index` per frame (e.g. from a batch index) so each frame
pulls its own zoom/angle/translation from the schedule.

### Getting the JSON out of the BFL dashboard

The dashboard's analysis object is defined in
`BFL-ui-revamp/ui/lib/audio-analysis.ts` (`AudioAnalysisResult`) and cached in
`localStorage`. Paste that object (the one with `markers` + `waveform`) into
`analysis_json`. *Follow-up:* add an explicit "Export analysis JSON" button in the
dashboard so this is one click.

### Notes / follow-ups

- `KoshiSchedule` emits type `KOSHI_SCHEDULE`, but `KoshiMotionEngine` expects
  `KOSHI_MOTION_SCHEDULE` — they don't currently connect. This node emits the
  latter directly; a tiny bridge/rename would reconcile the two existing nodes.
- `video` mode is implemented but untested where `opencv-python` is broken under
  NumPy 2.x; it raises a clear error and the other modes still work.
- Tests: `tests/test_audio_motion_schedule.py` (run with `python tests/test_audio_motion_schedule.py`).
