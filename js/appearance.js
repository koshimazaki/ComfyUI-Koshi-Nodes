/**
 * Koshi Nodes Appearance
 * Sets node colors based on category
 */

import { app } from "../../../scripts/app.js";

// Color schemes
const COLORS = {
    // Effects - Graphite
    effects: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // Motion/V2V - Graphite (same visual, different category)
    motion: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // Generators - Graphite
    generators: { color: "#1a1a1a", bgcolor: "#2d2d2d" },
    // SIDKIT - Orange brand
    sidkit: { color: "#FF9F43", bgcolor: "#1a1a1a" },
    // Utility - Graphite with subtle difference
    utility: { color: "#252525", bgcolor: "#333333" },
};

// Node to category mapping
const NODE_CATEGORIES = {
    // Effects
    "Koshi_Bloom": "effects",
    "Koshi_ChromaticAberration": "effects",
    "Koshi_Hologram": "effects",
    "Koshi_Scanlines": "effects",
    "Koshi_VideoGlitch": "effects",
    "Koshi_Glitch": "effects",
    "Koshi_Dither": "effects",
    "Koshi_DitheringFilter": "effects",

    // Generators (creates content)
    "Koshi_GlitchCandies": "generators",
    "Koshi_ShapeMorph": "generators",
    "Koshi_NoiseDisplace": "generators",
    "Koshi_Raymarcher": "generators",
    "KoshiDitheringRaymarcher": "generators",

    // SIDKIT
    "Koshi_Binary": "sidkit",
    "Koshi_Greyscale": "sidkit",
    "Koshi_OLEDScreen": "sidkit",
    "Koshi_OLEDPreview": "sidkit",
    "Koshi_PixelScaler": "sidkit",
    "Koshi_SpriteSheet": "sidkit",
    "Koshi_SIDKITScreen": "sidkit",
    "Koshi_XBMExport": "sidkit",

    // Motion & V2V (animation/video)
    "Koshi_ColorMatchLAB": "motion",
    "Koshi_OpticalFlowWarp": "motion",
    "Koshi_V2VProcessor": "motion",
    "Koshi_V2VMetadata": "motion",
    "Koshi_Schedule": "motion",
    "Koshi_ScheduleMulti": "motion",
    "Koshi_MotionEngine": "motion",
    "Koshi_MotionBatch": "motion",
    "Koshi_SemanticMotion": "motion",
    "Koshi_Feedback": "motion",
    "Koshi_FeedbackSimple": "motion",
    "Koshi_AnimationPipeline": "motion",
    "Koshi_FrameIterator": "motion",

    // Utility
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
