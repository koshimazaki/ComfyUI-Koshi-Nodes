```
██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗
██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝
█████╔╝ ██║   ██║███████╗███████║██║    ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗
██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║
██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```
░░█ **Custom nodes for ComfyUI** - Flux Motion V2V | Effects | Generators | Audio Conditioning | SIDKIT OLED █░░

---

# ComfyUI-Koshi-Nodes

**20 focused nodes** for ComfyUI: **Motion** (Deforum-style animation), **Effects** (unified effects + standalone bloom/glitch/chromatic), **Generators** (procedural patterns, raymarched 3D), **Audio / Conditioning** (AKUSPACE prompt and CLIP encoding), **SIDKIT** (OLED/embedded display export), and **Utility** (metadata capture).

> Consolidated core plus two AKUSPACE spatial-audio conditioning nodes — cleaner interfaces, fewer clicks.

https://github.com/user-attachments/assets/8c9f2c39-71c7-405a-bb94-10b0b9e96c32

*Glitch Candies shaders, OLED emulation and binary export.*

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

| Prefix | Category | Nodes |
|--------|----------|-------|
| `░▀░` | Effects | Koshi Effects (unified), Bloom, Chromatic, Glitch, Dither, Dithering Filter |
| `▄▀▄` | Motion | Schedule, Motion Engine, Feedback |
| `▄█▄` | Generators | Glitch Candies, Shape Morph, Noise Displace, Raymarcher |
| `░▒░` | SIDKIT/Export | SIDKIT Binary, Greyscale, SIDKIT OLED, Sprite Sheet |
| `◊` | Utility | Metadata |
| `◉` | Audio / Conditioning | AKUSPACE Prompt, AKUSPACE Text Encode |

## All 20 Nodes

### Effects (6 nodes)
Post-processing effects based on [alien.js](https://github.com/alienkitty/alien.js) and custom shaders.

| Node | Description |
|------|-------------|
| `░▀░ Koshi Effects` | **Unified** — 7 effects: dither, bloom, glitch, hologram, video glitch, scanlines, chromatic. Stack to combine. |
| `░▀░ KN Bloom` | Unreal-style bloom (GPU/CPU fallback) |
| `░▀░ KN Chromatic` | RGB channel separation |
| `░▀░ KN Glitch` | Shader-based glitch distortion |
| `░▀░ KN Dither` | Bayer, floyd-steinberg, atkinson, halftone |
| `░▀░ KN Dithering Filter` | GPU-accelerated dithering filter |

### Motion (3 nodes)
Deforum-inspired animation engine for FLUX models.

**Pipeline:**
```
▄▀▄ Schedule → ▄▀▄ Motion Engine → KSampler
                                      ↓
                              ▄▀▄ Feedback (loop)
```

| Node | Description |
|------|-------------|
| `▄▀▄ KN Schedule` | Parse keyframe strings (`0:(1.0), 30:(0.5)`) with interpolation and easing |
| `▄▀▄ KN Motion Engine` | Apply motion transforms to latents (zoom, angle, translation) |
| `▄▀▄ KN Feedback` | Frame-to-frame coherence: color match, sharpen, noise, auto-correct |

### Generators (4 nodes)
Procedural patterns, fractals, and raymarched 3D shapes.

| Node | Description |
|------|-------------|
| `▄█▄ Glitch Candies` | 27 patterns: waves, plasma, voronoi, fractals, raymarched 3D |
| `▄█▄ Shape Morph` | Blend/morph between two images with easing |
| `▄█▄ Noise Displace` | FBM noise displacement with animation |
| `▄█▄ Raymarcher` | Raymarched 3D shapes with dithering (requires ModernGL) |

**Patterns:**
- **2D:** waves, circles, plasma, voronoi, checkerboard, swirl, ripple
- **Fractals:** mandelbrot, julia, sierpinski
- **Glitch:** fbm_noise, cell_noise, distorted_grid, height_map, glitch_cubes
- **Raymarched 3D:** rm_cube, rm_sphere, rm_torus, rm_octahedron, rm_gyroid, rm_menger, rm_cylinder, rm_cone, rm_capsule, rm_pyramid

All patterns support `loop_frames` for seamless animation loops.

**3D Camera Controls:** Raymarched shapes have `rotation_x`, `rotation_y`, and `camera_distance` inputs for orbital control.

### SIDKIT / Export (4 nodes)
Nodes for [SIDKIT](https://sidkit.pages.dev/) synthesizer OLED displays (SSD1306, SSD1363) and export.

| Node | Description |
|------|-------------|
| `░▒░ KN SIDKIT Binary` | Threshold (simple, adaptive, otsu, dither) + hex/XBM/C header export |
| `░▒░ KN Greyscale` | Greyscale with bit depth quantization (1/2/4/8-bit) |
| `░▒░ KN SIDKIT OLED` | OLED display viewer with WebGL preview and screen presets |
| `░▒░ KN Sprite Sheet` | Combine frames into sprite sheet grid |

**OLED Presets:** SSD1306 128x64, SSD1306 128x32, SSD1363 256x128, custom

**Bit Depths:** 1-bit (2), 2-bit (4), 4-bit (16), 8-bit (256)

### Utility (1 node)
| Node | Description |
|------|-------------|
| `◊ Koshi Metadata` | **Unified metadata node** — capture workflow settings, display, and save as JSON/text. Extracts seed, steps, cfg, model, LoRAs, prompts. |

### Audio / Conditioning (2 nodes)

AKUSPACE provides a shared WebGL spatial controller for an experimental LTX
audio LoRA. The LoRA checkpoint is external and is not bundled with this node
pack. See the [AKUSPACE guide](./docs/AKUSPACE.md) and
[interactive project page](https://audiolora.dev/).

| Node | Description |
|------|-------------|
| `◉ AKUSPACE Prompt` | Modular Prompt → Prompt treatment node, compatible with camera and other string controls. |
| `◉ AKUSPACE Text Encode` | CLIP + editable Text → Conditioning convenience node with live combined-prompt preview. |

## Project Structure

```
ComfyUI-Koshi-Nodes/
├── nodes/
│   ├── effects/        # Koshi Effects (unified), bloom, glitch, chromatic aberration, raymarcher
│   ├── export/         # SIDKIT OLED, sprite sheet
│   ├── flux_motion/    # Schedule, motion engine, feedback
│   │   └── core/       # Interpolation, easing, transforms
│   ├── generators/     # Glitch Candies, shape morph, noise displace, raymarcher
│   ├── utility/        # Metadata (unified capture/display/save)
│   ├── utils/          # Shared utilities (tensor ops, metadata)
│   ├── audio/          # AKUSPACE prompt and CLIP-conditioning nodes
│   └── image/          # SIDKIT Edition
│       ├── binary/     # Threshold + hex export
│       ├── dither/     # Bayer, Floyd-Steinberg, Atkinson, Halftone
│       └── greyscale/  # Quantization, algorithms
├── shaders/            # GLSL shaders (bloom, chromatic aberration)
├── workflows/          # Example workflow JSONs
└── js/                 # Live preview, orbital controls, Nodes 2.0 theme
```

## Live Preview & WebGL

Most nodes include **inline live preview** directly in the node UI - no Preview Image node needed.

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

**Motion Animation:**
```
▄▀▄ Schedule → ▄▀▄ Motion Engine → KSampler
                                      ↓
                              ▄▀▄ Feedback (loop)
```

**Stacked Effects:**
```
Image → ░▀░ Koshi Effects (hologram) → ░▀░ Koshi Effects (chromatic) → ░▀░ KN Bloom
```

**SIDKIT OLED Export:**
```
Image → ░▒░ KN Greyscale → ░▀░ KN Dither → ░▒░ KN SIDKIT OLED → ░▒░ KN Sprite Sheet
```

**Generator → Effect → Export:**
```
▄█▄ Glitch Candies → ░▀░ Koshi Effects (scanlines) → ░▒░ KN Sprite Sheet
```

**AKUSPACE modular prompt:**
```
Base Prompt → ◉ AKUSPACE Prompt → CLIP Text Encode → Conditioning
```

**AKUSPACE integrated encode:**
```
LoRA-patched CLIP + Text → ◉ AKUSPACE Text Encode → Conditioning
```

## Dependencies

**Required:** `torch`, `numpy`, `pillow`

**Optional:**
- `moderngl` - GPU effects
- `scipy` - Better dithering/edge detection
- `opencv-python` - Color matching in Feedback node (LAB space)

## Credits

- Generated using [VibeComfy Tools](https://github.com/peteromallet/VibeComfy)
- Uses [FLUX](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [alien.js](https://github.com/alienkitty/alien.js) by Patrick Schroen - Chromatic aberration, bloom shaders (MIT)
- Hologram based on [CreaturesSite](https://github.com/koshimazaki/CreaturesSite)
- SIDKIT Edition for [SIDKIT Synthesizer](https://github.com/koshimazaki/SIDKIT)
- AKUSPACE WebGL controller uses Vue and Three.js; see [third-party notices](./THIRD_PARTY_NOTICES.md)
