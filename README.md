```
██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗
██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝
█████╔╝ ██║   ██║███████╗███████║██║    ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗
██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║
██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```
░░█ **Custom nodes for ComfyUI** - Flux Motion V2V | SIDKIT OLED | Post-FX | Generators █░░

---

# ComfyUI-Koshi-Nodes

Custom nodes for ComfyUI: **Flux Motion** (Deforum-inspired V2V), **SIDKIT Edition** (OLED/embedded display), and **Post-FX** (hologram, bloom, glitch).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git Koshi-Nodes
cd Koshi-Nodes && pip install -r requirements.txt
```

## Quick Setup (RunPod / Full Install)

One-line setup with ComfyUI + FLUX models + Koshi Nodes:

```bash
curl -sL https://raw.githubusercontent.com/koshimazaki/ComfyUI-Koshi-Nodes/main/setup_comfyui_flux.sh | bash
```

Or with model preset flags:

```bash
# Interactive menu (default)
./setup_comfyui_flux.sh

# Presets - skip menu
./setup_comfyui_flux.sh --minimal      # Schnell + FP8 T5 (~17GB)
./setup_comfyui_flux.sh --full         # Schnell + Dev + FP16 T5 (~46GB)
./setup_comfyui_flux.sh --fp8          # FP8 optimized (~17GB, lower VRAM)
./setup_comfyui_flux.sh --skip-models  # No model downloads

# With HuggingFace token (required for FLUX models)
./setup_comfyui_flux.sh --minimal --token=hf_yourtoken
# Or via environment variable
HF_TOKEN=hf_yourtoken ./setup_comfyui_flux.sh --minimal
```

> **Note:** FLUX models require HuggingFace authentication. Get your token at https://huggingface.co/settings/tokens

**Model Presets:**

| Preset | Models | Size | VRAM | Use Case |
|--------|--------|------|------|----------|
| `--minimal` | Schnell + FP8 T5 | ~17GB | 16GB+ | Fast generation, testing |
| `--full` | Schnell + Dev + FP16 T5 | ~46GB | 24GB+ | Best quality |
| `--fp8` | Dev FP8 + FP8 T5 | ~17GB | 12GB+ | Memory efficient |

**Included:**
- ComfyUI (latest)
- Koshi Nodes
- ComfyUI-Manager
- VideoHelperSuite
- FLUX models (based on preset)

## Node Naming

Nodes are prefixed by category for easy identification:

| Prefix | Category | Description |
|--------|----------|-------------|
| `▄▀▄ KN` | Effects & Processing | V2V, color match, hologram, bloom, glitch, utility |
| `▄█▄ KN` | Generators | Procedural patterns, fractals, raymarched 3D |
| `░▒░ KN` | SIDKIT | OLED display, dithering, binary, export |

## Node Categories

### Flux Motion (V2V)
Deforum-inspired temporal coherence for video stylization with FLUX models.

| Node | Description |
|------|-------------|
| `▄▀▄ KN V2V Processor` | Main V2V pipeline - 4 modes: pure, temporal, motion, ultimate |
| `▄▀▄ KN Color Match LAB` | Match colors to anchor frame (LAB space) |
| `▄▀▄ KN Optical Flow Warp` | Warp frames using optical flow |

**V2V Modes:**
| Mode | Technique |
|------|-----------|
| `pure` | Frame-by-frame + LAB color match |
| `temporal` | Blend previous output with current input |
| `motion` | Warp previous output via optical flow |
| `ultimate` | All techniques combined |

### Generators
Procedural patterns, fractals, and raymarched 3D shapes. Based on SIDKIT shader system.

| Node | Description |
|------|-------------|
| `▄█▄ KN Glitch Candies` | 22 patterns: waves, plasma, voronoi, fractals, raymarched 3D |
| `▄█▄ KN Shape Morph` | Blend/morph between two patterns with easing |
| `▄█▄ KN Noise Displace` | FBM noise displacement with animation |

**Patterns:**
- **2D:** waves, circles, plasma, voronoi, checkerboard, swirl, ripple
- **Fractals:** mandelbrot, julia, sierpinski
- **Glitch Candies:** fbm_noise, cell_noise, distorted_grid, height_map
- **Raymarched 3D:** rm_cube, rm_sphere, rm_torus, rm_octahedron, rm_gyroid, rm_menger

All patterns support `loop_frames` for seamless animation loops.

**3D Camera Controls:** Raymarched shapes have `rotation_x`, `rotation_y`, and `camera_distance` inputs for orbital control.

### Post-FX Effects
Post-processing effects based on [alien.js](https://github.com/alienkitty/alien.js) and custom shaders.

| Node | Description |
|------|-------------|
| `▄▀▄ KN Hologram` | Full hologram (scanlines, glitch, edge glow, grid, color tint) |
| `▄▀▄ KN Scanlines` | Horizontal/vertical scanlines |
| `▄▀▄ KN Video Glitch` | RGB split glitch distortion |
| `▄▀▄ KN Chromatic Aberration` | RGB channel separation |
| `▄▀▄ KN Bloom` | Unreal-style bloom (GPU/CPU fallback) |

**Hologram presets:** cyan, red_error, green_matrix, purple, orange, white

### Utility
Metadata capture, workflow settings, and helper nodes.

| Node | Description |
|------|-------------|
| `▄▀▄ KN Capture Settings` | Extract all workflow settings as JSON (seed, steps, model, prompts) |
| `▄▀▄ KN Save Metadata` | Save metadata JSON to file with timestamp |
| `▄▀▄ KN Display Metadata` | Display metadata in UI |

**Captured params:** seed, steps, cfg, sampler, scheduler, model, LoRAs, positive/negative prompts, dimensions

### SIDKIT Edition
Nodes optimized for [SIDKIT](https://github.com/koshimazaki/SIDKIT) synthesizer OLED displays (SSD1306, SSD1363).

| Node | Description |
|------|-------------|
| `░▒░ KN Dither` | All dithering: bayer, floyd-steinberg, atkinson, halftone |
| `░▒░ KN Binary` | Threshold methods + hex export for C headers |
| `░▒░ KN Greyscale` | Greyscale conversion with bit depth quantization |
| `░▒░ KN OLED Screen` | OLED emulator with region presets |
| `░▒░ KN Pixel Scaler` | Lanczos/nearest scaling to OLED resolutions |
| `░▒░ KN Sprite Sheet` | Combine frames into sprite grid |
| `░▒░ KN SIDKIT Screen` | Export to .sidv/.xbm/.h for Teensy |

**OLED Presets:** SSD1306 128x64, SSD1363 256x128, custom

**Region Presets:** full, left_half (128x128), right_half, quadrants (64x64)

**Bit Depths:** 1-bit (2), 2-bit (4), 4-bit (16), 8-bit (256)

## Project Structure

```
ComfyUI-Koshi-Nodes/
├── nodes/
│   ├── effects/        # Hologram, bloom, glitch, chromatic aberration
│   ├── export/         # OLED screen, SIDKIT export, sprite sheets
│   ├── flux_motion/    # V2V processor, optical flow, color match
│   │   └── core/       # Interpolation, easing, transforms
│   ├── generators/     # Glitch Candies, shape morph, noise displace
│   ├── utility/        # Metadata capture, settings save
│   └── image/          # SIDKIT Edition
│       ├── binary/     # Threshold + hex export
│       ├── dither/     # Bayer, Floyd-Steinberg, Atkinson, Halftone
│       └── greyscale/  # Quantization, algorithms
├── shaders/            # GLSL shaders (bloom, chromatic aberration)
├── workflows/          # Example workflow JSONs
└── web/js/             # Live preview + orbital controls
```

## Live Preview

All effect, generator, and SIDKIT nodes include **inline live preview** - see results directly in the node without connecting to a Preview Image node.

**Features:**
- Toggle preview on/off per node
- Batch preview (shows first 4 frames)
- Pixel-perfect rendering for OLED preview

**3D Orbital Controls** (Generators with `rm_*` patterns):
- **Drag** image to rotate (updates `rotation_x`/`rotation_y`)
- **Scroll** to zoom (updates `camera_distance`)
- **Reset View** button to return to default

## Example Workflows

In `workflows/`:
- `koshi_v2v_ultimate.json` - Full V2V with motion + temporal + color match
- `koshi_oled_sprite_pipeline.json` - Image → dither → OLED preview → export
- `koshi_sprite_sheet.json` - Video → sprite sheet for game dev

## Quick Pipelines

**V2V Stylization:**
```
Video → ▄▀▄ KN V2V Processor (ultimate) → VHS_VideoCombine
```

**Hologram Effect:**
```
Image → ▄▀▄ KN Hologram (cyan) → ▄▀▄ KN Chromatic Aberration → ▄▀▄ KN Bloom
```

**SIDKIT OLED Export:**
```
Image → ░▒░ KN Pixel Scaler (128x64) → ░▒░ KN Dither (bayer, 2 levels)
      → ░▒░ KN OLED Screen (preview) → ░▒░ KN SIDKIT Screen (.xbm)
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
