# Project structure

The repository root keeps only package metadata, installation essentials, and
the main README. Detailed guides and operational tooling live in focused
subdirectories.

```text
ComfyUI-Koshi-Nodes/
├── nodes/
│   ├── audio/          # Audio → Motion plus the three AKUSPACE nodes
│   ├── effects/        # Unified effects, bloom, glitch, chromatic
│   ├── flux_motion/    # Schedule, motion engine, feedback
│   ├── generators/     # Procedural patterns and raymarching
│   ├── image/          # Dither, greyscale, and binary export
│   ├── utility/        # Metadata
│   └── utils/          # Shared tensor, preview, and metadata helpers
├── docs/               # AKUSPACE, workflow, structure, and licence guides
├── js/                 # Live previews, controls, and the AKUSPACE widget
├── scripts/
│   ├── setup_comfyui_flux.sh
│   └── akuspace/       # LTX box setup, A/B, batch, and verification tools
├── shaders/            # GLSL effects
├── tests/              # Unit, integration, and registration tests
└── workflows/
    └── akuspace/       # Public LTX-2.5 API workflows
```

See [`scripts/README.md`](../scripts/README.md) for operational commands and
[`WORKFLOWS.md`](./WORKFLOWS.md) for the example graph catalogue.
