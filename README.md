```
██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗
██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝
█████╔╝ ██║   ██║███████╗███████║██║    ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗
██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║
██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```
░░█ **Custom nodes for ComfyUI** - Flux Motion V2V | Effects | Generators | SIDKIT OLED █░░

---

# ComfyUI-Koshi-Nodes

Custom nodes for ComfyUI: **Flux Motion** (Deforum-inspired animation & V2V), **Effects** (hologram, bloom, glitch, dither), **Generators** (procedural patterns, raymarched 3D), and **SIDKIT Edition** (OLED/embedded display export).

## Installation

**One-liner** (existing ComfyUI):
```bash
cd ComfyUI/custom_nodes && git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git Koshi-Nodes && pip install -r Koshi-Nodes/requirements.txt
```

**Or step by step:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git Koshi-Nodes
cd Koshi-Nodes && pip install -r requirements.txt
```

## Quick Setup (RunPod)

Full setup with ComfyUI + FLUX Q4 GGUF + Koshi Nodes (~6GB VRAM):

```bash
curl -sL https://raw.githubusercontent.com/koshimazaki/ComfyUI-Koshi-Nodes/main/setup_comfyui_flux.sh | bash -s -- --runpod --gguf
```

Includes: ComfyUI, Koshi Nodes, ComfyUI-Manager, VideoHelperSuite, ComfyUI-GGUF, FLUX Dev Q4

## Node Naming

Nodes are prefixed by category for easy identification:

| Prefix | Category | Description |
|--------|----------|-------------|
| `░▀░ KN` | Effects | Hologram, bloom, glitch, chromatic aberration, dither |
| `▄▀▄ KN` | Motion | V2V, color match, optical flow, semantic motion |
| `▀▄▀ KN` | Motion Core | Schedule, motion engine, feedback |
| `▄█▄ KN` | Generators | Procedural patterns, fractals, raymarched 3D |
| `░▒░ KN` | SIDKIT | OLED display, binary, greyscale, export |
| `◊ KN` | Utility | Metadata capture, settings save |

## Node Categories

### Flux Motion (V2V)
Deforum-inspired animation engine and V2V processing for FLUX models.

**Modular pipeline** (recommended):
```
▀▄▀ Schedule → ▀▄▀ Multi-Schedule → ▀▄▀ Motion Engine → KSampler
                                                      ↓
                                              ▀▄▀ Feedback (loop)
```

| Node | Description |
|------|-------------|
| `▀▄▀ KN Schedule Parser` | Parse Deforum-style keyframe strings (`0:(1.0), 30:(0.5)`) |
| `▀▄▀ KN Multi-Schedule` | Combine multiple schedules (zoom, angle, translation) |
| `▀▄▀ KN Motion Engine` | Apply motion vectors and transforms to latents |
| `▀▄▀ KN Motion Batch` | Batch process motion across frame sequences |
| `▀▄▀ KN Feedback` | Frame-to-frame coherence with color matching and enhancement |
| `▀▄▀ KN Feedback Simple` | Lightweight feedback for quick iteration |
| `▄▀▄ KN Semantic Motion` | Generate motion from text descriptions ("slow zoom in, pan left") |
| `▄▀▄ KN Color Match LAB` | Match colors to anchor frame (LAB space) |
| `▄▀▄ KN Optical Flow Warp` | Warp frames using optical flow |
| `▄▀▄ KN Image Blend` | Blend images with configurable strength |
| `▄▀▄ KN Frame Iterator` | Iterate frames for animation loops |
| `▄▀▄ KN V2V Metadata` | Save V2V processing metadata alongside output |

**Semantic Motion Presets:** zoom in/out, pan left/right/up/down, rotate, dolly, orbit, push/pull, spin, static

### Effects
Post-processing effects based on [alien.js](https://github.com/alienkitty/alien.js) and custom shaders.

| Node | Description |
|------|-------------|
| `░▀░ KN Hologram` | Full hologram (scanlines, glitch, edge glow, grid, color tint) |
| `░▀░ KN Scanlines` | Horizontal/vertical scanlines |
| `░▀░ KN Video Glitch` | RGB split glitch distortion |
| `░▀░ KN Chromatic Aberration` | RGB channel separation |
| `░▀░ KN Bloom` | Unreal-style bloom (GPU/CPU fallback) |
| `░▀░ KN Glitch` | Shader-based glitch distortion |
| `░▀░ KN Dither` | All dithering: bayer, floyd-steinberg, atkinson, halftone |
| `░▀░ KN Dithering Filter` | GPU-accelerated dithering filter |

**Hologram presets:** cyan, red_error, green_matrix, purple, orange, white

### Generators
Procedural patterns, fractals, and raymarched 3D shapes. Based on SIDKIT shader system.

| Node | Description |
|------|-------------|
| `▄█▄ KN Glitch Candies` | 22 patterns: waves, plasma, voronoi, fractals, raymarched 3D |
| `▄█▄ KN Shape Morph` | Blend/morph between two patterns with easing |
| `▄█▄ KN Noise Displace` | FBM noise displacement with animation |
| `▄█▄ KN Raymarcher` | Dedicated raymarched 3D shapes with dithering |

**Patterns:**
- **2D:** waves, circles, plasma, voronoi, checkerboard, swirl, ripple
- **Fractals:** mandelbrot, julia, sierpinski
- **Glitch Candies:** fbm_noise, cell_noise, distorted_grid, height_map
- **Raymarched 3D:** rm_cube, rm_sphere, rm_torus, rm_octahedron, rm_gyroid, rm_menger

All patterns support `loop_frames` for seamless animation loops.

**3D Camera Controls:** Raymarched shapes have `rotation_x`, `rotation_y`, and `camera_distance` inputs for orbital control.

### Utility
Metadata capture, workflow settings, and helper nodes.

| Node | Description |
|------|-------------|
| `◊ KN Capture Settings` | Extract all workflow settings as JSON (seed, steps, model, prompts) |
| `◊ KN Save Metadata` | Save metadata JSON to file with timestamp |
| `◊ KN Display Metadata` | Display metadata in UI |

**Captured params:** seed, steps, cfg, sampler, scheduler, model, LoRAs, positive/negative prompts, dimensions

### SIDKIT Edition
Nodes optimized for [SIDKIT](https://sidkit.pages.dev/) synthesizer OLED displays (SSD1306, SSD1363).

| Node | Description |
|------|-------------|
| `░▒░ KN Binary` | Threshold methods + hex export for C headers |
| `░▒░ KN Greyscale` | Greyscale conversion with bit depth quantization |
| `░▒░ KN OLED Screen` | OLED emulator with region presets |
| `░▒░ KN OLED Preview` | Inline OLED preview in node |
| `░▒░ KN Pixel Scaler` | Lanczos/nearest scaling to OLED resolutions |
| `░▒░ KN Sprite Sheet` | Combine frames into sprite grid |
| `░▒░ KN XBM Export` | Export frames as XBM format |
| `░▒░ KN SIDKIT Screen` | Export to .sidv/.xbm/.h for Teensy |

**OLED Presets:** SSD1306 128x64, SSD1363 256x128, custom

**Region Presets:** full, left_half (128x128), right_half, quadrants (64x64)

**Bit Depths:** 1-bit (2), 2-bit (4), 4-bit (16), 8-bit (256)

## Project Structure

```
ComfyUI-Koshi-Nodes/
├── nodes/
│   ├── effects/        # Hologram, bloom, glitch, chromatic aberration
│   ├── export/         # OLED screen, SIDKIT export, sprite sheets, XBM
│   ├── flux_motion/    # Motion engine, schedule, feedback, V2V, semantic
│   │   └── core/       # Interpolation, easing, transforms
│   ├── generators/     # Glitch Candies, shape morph, noise displace, raymarcher
│   ├── utility/        # Metadata capture, settings save
│   ├── utils/          # Shared utilities (tensor ops, metadata)
│   ├── audio/          # (Reserved for future audio-reactive nodes)
│   └── image/          # SIDKIT Edition
│       ├── binary/     # Threshold + hex export
│       ├── dither/     # Bayer, Floyd-Steinberg, Atkinson, Halftone
│       └── greyscale/  # Quantization, algorithms
├── shaders/            # GLSL shaders (bloom, chromatic aberration)
├── workflows/          # Example workflow JSONs
└── js/                 # Live preview + orbital controls
```

## Live Preview & WebGL

31 nodes include **inline live preview** directly in the node UI - no Preview Image node needed.

- Toggle preview on/off per node
- Batch preview (first 4 frames, "+N more" indicator)
- Collapsible preview widget (▼/▶)

**WebGL Shaders** (9 built-in): passthrough, bayer dither, floyd-steinberg, halftone, glitch, bloom, threshold, greyscale, OLED emulation.

**OLED Emulation Shader:** Exact SIDKIT Bayer matrices (2x2/4x4/8x8), 1/2/4/8-bit depth, 6 color modes (grayscale, green, blue, amber, white, yellow), pixel gap/grid effect, bloom glow.

**3D Orbital Controls** (Generators with `rm_*` patterns):
- **Drag** to rotate (updates `rotation_x`/`rotation_y`, clamped ±90°)
- **Scroll** to zoom (`camera_distance`, range 1-10)
- **Reset View** button to restore defaults

## Example Workflows

In `workflows/`:
- `koshi_v2v_ultimate.json` - Full V2V with motion + temporal + color match
- `koshi_v2v_complete.json` - Complete V2V pipeline
- `koshi_v2v_motion.json` - Motion-focused V2V
- `koshi_v2v_temporal.json` - Temporal coherence V2V
- `koshi_v2v_pure.json` - Minimal V2V setup
- `koshi_oled_sprite_pipeline.json` - Image → dither → OLED preview → export
- `koshi_sprite_sheet.json` - Video → sprite sheet for game dev

## Quick Pipelines

**Motion Animation (modular):**
```
▀▄▀ Schedule → ▀▄▀ Multi-Schedule → ▀▄▀ Motion Engine → KSampler
                                                      ↓
                                              ▀▄▀ Feedback (loop)
```

**Semantic Motion:**
```
▄▀▄ Semantic Motion ("zoom in, pan left") → ▀▄▀ Motion Engine → KSampler
```

**Hologram Effect:**
```
Image → ░▀░ Hologram → ░▀░ Chromatic Aberration → ░▀░ Bloom
```

**SIDKIT OLED Export:**
```
Image → ░▒░ Pixel Scaler → ░▀░ Dither → ░▒░ OLED Screen → ░▒░ SIDKIT Screen
```

## Dependencies

**Required:** `torch`, `numpy`, `pillow`

**Optional:**
- `moderngl` - GPU effects
- `scipy` - Better dithering/edge detection
- `opencv-python` - Optical flow for V2V motion mode

## Credits

- Generated using [VibeComfy Tools](https://github.com/peteromallet/VibeComfy)
- Uses [FLUX](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [alien.js](https://github.com/alienkitty/alien.js) by Patrick Schroen - Chromatic aberration, bloom shaders (MIT)
- Hologram based on [CreaturesSite](https://github.com/koshimazaki/CreaturesSite)
- SIDKIT Edition for [SIDKIT Synthesizer](https://github.com/koshimazaki/SIDKIT)
