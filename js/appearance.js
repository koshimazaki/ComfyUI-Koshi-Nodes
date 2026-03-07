/**
 * Koshi Nodes Appearance
 * Sets node colors based on category
 *
 * Groups (18 nodes):
 * - Effects (4):      ░▀░ KN — Bloom, Chromatic Aberration, Glitch, Effects
 * - Generators (4):   ▄█▄    — Glitch Candies, Shape Morph, Noise Displace, Raymarcher
 * - Flux Motion (3):  ▄▀▄ KN — Schedule, Motion Engine, Feedback
 * - SIDKIT (6):       ░▒░ KN — OLED Screen, Sprite Sheet, Binary, Greyscale, Dither, Dithering Filter
 * - Utility (1):      ◊      — Metadata
 */

import { app } from "../../../scripts/app.js";

// Color schemes
const COLORS = {
    // Effects — white on graphite (░▀░)
    effects: { color: "#ffffff", bgcolor: "#2d2d2d" },
    // Flux Motion — white on graphite (▄▀▄)
    motion: { color: "#ffffff", bgcolor: "#2d2d2d" },
    // Generators — white on graphite (▄█▄)
    generators: { color: "#ffffff", bgcolor: "#2d2d2d" },
    // SIDKIT/Export — black text on orange/white (░▒░)
    sidkit: { color: "#FF9F43", bgcolor: "#ffffff" },
    // Utility — white on subtle graphite (◊)
    utility: { color: "#ffffff", bgcolor: "#333333" },
};

// Node to category mapping
const NODE_CATEGORIES = {
    // Effects (░▀░)
    "Koshi_Effects": "effects",
    "Koshi_Bloom": "effects",
    "Koshi_ChromaticAberration": "effects",
    "Koshi_Glitch": "effects",

    // Generators (▄█▄)
    "Koshi_GlitchCandies": "generators",
    "Koshi_ShapeMorph": "generators",
    "Koshi_NoiseDisplace": "generators",
    "Koshi_Raymarcher": "generators",

    // Flux Motion (▄▀▄)
    "Koshi_Schedule": "motion",
    "Koshi_MotionEngine": "motion",
    "Koshi_Feedback": "motion",

    // SIDKIT/Export (░▒░)
    "Koshi_OLEDScreen": "sidkit",
    "Koshi_SpriteSheet": "sidkit",
    "Koshi_Binary": "sidkit",
    "Koshi_Greyscale": "sidkit",
    "Koshi_Dither": "sidkit",
    "Koshi_DitheringFilter": "sidkit",

    // Utility (◊)
    "Koshi_Metadata": "utility",
};

app.registerExtension({
    name: "Koshi-Nodes.appearance",
    nodeCreated(node) {
        const category = NODE_CATEGORIES[node.comfyClass];
        if (category && COLORS[category]) {
            node.color = COLORS[category].color;
            node.bgcolor = COLORS[category].bgcolor;
        }
    }
});
