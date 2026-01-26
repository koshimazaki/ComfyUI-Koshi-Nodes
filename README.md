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
git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes
pip install -r requirements.txt
```

## Node Categories

### Flux Motion (V2V)
Deforum-inspired temporal coherence for video stylization with FLUX models.

| Node | Description |
|------|-------------|
| `Koshi_V2VProcessor` | Main V2V pipeline - 4 modes: pure, temporal, motion, ultimate |
| `Koshi_ColorMatchLAB` | Match colors to anchor frame (LAB space) |
| `Koshi_OpticalFlowWarp` | Warp frames using optical flow |

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
| `Koshi_GlitchCandies` | 22 patterns: waves, plasma, voronoi, fractals, raymarched 3D |
| `Koshi_ShapeMorph` | Blend/morph between two patterns with easing |
| `Koshi_NoiseDisplace` | FBM noise displacement with animation |

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
| `Koshi_Hologram` | Full hologram (scanlines, glitch, edge glow, grid, color tint) |
| `Koshi_Scanlines` | Horizontal/vertical scanlines |
| `Koshi_VideoGlitch` | RGB split glitch distortion |
| `Koshi_ChromaticAberration` | RGB channel separation |
| `Koshi_Bloom` | Unreal-style bloom (GPU/CPU fallback) |

**Hologram presets:** cyan, red_error, green_matrix, purple, orange, white

### Utility
Metadata capture, workflow settings, and helper nodes.

| Node | Description |
|------|-------------|
| `Koshi_CaptureSettings` | Extract all workflow settings as JSON (seed, steps, model, prompts) |
| `Koshi_SaveMetadata` | Save metadata JSON to file with timestamp |
| `Koshi_DisplayMetadata` | Display metadata in UI |

**Captured params:** seed, steps, cfg, sampler, scheduler, model, LoRAs, positive/negative prompts, dimensions

### SIDKIT Edition
Nodes optimized for [SIDKIT](https://github.com/koshimazaki/SIDKIT) synthesizer OLED displays (SSD1306, SSD1363).

| Node | Description |
|------|-------------|
| `Koshi_Dither` | All dithering: bayer, floyd-steinberg, atkinson, halftone |
| `Koshi_Binary` | Threshold methods + hex export for C headers |
| `Koshi_Greyscale` | Greyscale conversion with bit depth quantization |
| `Koshi_OLEDScreen` | OLED emulator with region presets |
| `Koshi_PixelScaler` | Lanczos/nearest scaling to OLED resolutions |
| `Koshi_SpriteSheet` | Combine frames into sprite grid |
| `SIDKIT Screen` | Export to .sidv/.xbm/.h for Teensy |

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
Video → Koshi_V2VProcessor (ultimate) → VHS_VideoCombine
```

**Hologram Effect:**
```
Image → Koshi_Hologram (cyan) → Koshi_ChromaticAberration → Koshi_Bloom
```

**SIDKIT OLED Export:**
```
Image → Koshi_PixelScaler (128x64) → Koshi_Dither (bayer, 2 levels)
      → Koshi_OLEDScreen (preview) → SIDKIT Screen (.xbm)
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
