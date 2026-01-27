/**
 * Koshi Nodes Appearance
 * Sets node colors based on category
 *
 * Prefixes:
 * - Effects: ░▀░ KN (light, symmetric)
 * - Motion/V2V: ▄▀▄ KN (wave pattern)
 * - Generators: ▄█▄ KN (solid)
 * - SIDKIT: ░▒░ KN (orange brand)
 * - Utility: ◊ KN (diamond)
 */

import { app } from "../../../scripts/app.js";

// Color schemes
const COLORS = {
    // Effects - Graphite (░▀░)
    effects: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // Motion/V2V - Graphite (▄▀▄)
    motion: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // Generators - Graphite (▄█▄)
    generators: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // SIDKIT/Export - Orange brand (░▒░)
    sidkit: { color: "#FF9F43", bgcolor: "#1a1a1a" },
    // Utility - Subtle graphite (◊)
    utility: { color: "#252525", bgcolor: "#333333" },
    // Deprecated - Reddish tint
    deprecated: { color: "#4a1a1a", bgcolor: "#2d2d2d" },
};

// Node to category mapping
const NODE_CATEGORIES = {
    // Effects (░▀░)
    "Koshi_Bloom": "effects",
    "KoshiBloomShader": "effects",
    "Koshi_ChromaticAberration": "effects",
    "Koshi_Hologram": "effects",
    "Koshi_Scanlines": "effects",
    "Koshi_VideoGlitch": "effects",
    "Koshi_Glitch": "effects",
    "Koshi_Dither": "effects",
    "Koshi_DitheringFilter": "effects",

    // Generators (▄█▄) - creates content
    "Koshi_GlitchCandies": "generators",
    "Koshi_ShapeMorph": "generators",
    "Koshi_NoiseDisplace": "generators",
    "Koshi_Raymarcher": "generators",
    "KoshiDitheringRaymarcher": "generators",

    // Motion & V2V (▄▀▄) - animation/video
    "Koshi_ColorMatchLAB": "motion",
    "Koshi_OpticalFlowWarp": "motion",
    "Koshi_ImageBlend": "motion",
    "Koshi_V2VMetadata": "motion",
    "Koshi_Schedule": "motion",
    "Koshi_ScheduleMulti": "motion",
    "Koshi_MotionEngine": "motion",
    "Koshi_MotionBatch": "motion",
    "Koshi_SemanticMotion": "motion",
    "Koshi_Feedback": "motion",
    "Koshi_FeedbackSimple": "motion",
    "Koshi_FrameIterator": "motion",

    // Deprecated nodes (reddish)
    "Koshi_V2VProcessor": "deprecated",
    "Koshi_AnimationPipeline": "deprecated",

    // SIDKIT/Export (░▒░) - orange brand
    "Koshi_Binary": "sidkit",
    "Koshi_Greyscale": "sidkit",
    "Koshi_OLEDScreen": "sidkit",
    "Koshi_OLEDPreview": "sidkit",
    "Koshi_PixelScaler": "sidkit",
    "Koshi_SpriteSheet": "sidkit",
    "Koshi_SIDKITScreen": "sidkit",
    "Koshi_XBMExport": "sidkit",

    // Utility (◊)
    "Koshi_CaptureSettings": "utility",
    "Koshi_SaveMetadata": "utility",
    "Koshi_DisplayMetadata": "utility",
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
