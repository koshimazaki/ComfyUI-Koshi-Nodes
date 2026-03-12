/**
 * Koshi Nodes Appearance
 * Sets node colors based on category
 * Orange + Graphite theme for Nodes 2.0 (Vue) and LiteGraph canvas
 *
 * Groups (18 nodes):
 * - Effects (4):      ░▀░ KN — Bloom, Chromatic Aberration, Glitch, Effects
 * - Generators (4):   ▄█▄    — Glitch Candies, Shape Morph, Noise Displace, Raymarcher
 * - Flux Motion (3):  ▄▀▄ KN — Schedule, Motion Engine, Feedback
 * - SIDKIT (6):       ░▒░ KN — SIDKIT OLED, Sprite Sheet, Binary, Greyscale, Dither, Dithering Filter
 * - Utility (1):      ◊      — Metadata
 */

import { app } from "../../../scripts/app.js";

// Color schemes — orange title, black body (SIDKIT = white body)
const COLORS = {
    effects: { color: "#FF9F43", bgcolor: "#1a1a1a" },
    motion: { color: "#FF9F43", bgcolor: "#1a1a1a" },
    generators: { color: "#FF9F43", bgcolor: "#1a1a1a" },
    sidkit: { color: "#FF9F43", bgcolor: "#ffffff" },
    utility: { color: "#FF9F43", bgcolor: "#1a1a1a" },
};

// Node to category mapping
const NODE_CATEGORIES = {
    "Koshi_Effects": "effects",
    "Koshi_Bloom": "effects",
    "Koshi_ChromaticAberration": "effects",
    "Koshi_Glitch": "effects",
    "Koshi_GlitchCandies": "generators",
    "Koshi_ShapeMorph": "generators",
    "Koshi_NoiseDisplace": "generators",
    "Koshi_Raymarcher": "generators",
    "Koshi_Schedule": "motion",
    "Koshi_MotionEngine": "motion",
    "Koshi_Feedback": "motion",
    "Koshi_OLEDScreen": "sidkit",
    "Koshi_SpriteSheet": "sidkit",
    "Koshi_Binary": "sidkit",
    "Koshi_Greyscale": "sidkit",
    "Koshi_Dither": "sidkit",
    "Koshi_DitheringFilter": "sidkit",
    "Koshi_Metadata": "utility",
};

// Default node width — ~50% wider than ComfyUI default 225px
const DEFAULT_WIDTH = 380;

// Which widgets are visible per effect type
const EFFECT_WIDGETS = {
    dither: ["dither_method", "dither_levels"],
    bloom: ["bloom_threshold", "bloom_radius"],
    glitch: ["rgb_shift", "shake_amount", "noise_amount", "scan_lines"],
    hologram: ["holo_color", "edge_glow", "grid_opacity", "scan_lines"],
    video_glitch: [],
    scanlines: ["scanline_count", "scanline_direction"],
    chromatic: ["red_offset", "blue_offset"],
};

const ALWAYS_VISIBLE = [
    "image", "effect_type", "enabled", "mix", "intensity", "time", "seed",
];

function updateEffectWidgets(node) {
    const effectWidget = node.widgets?.find(w => w.name === "effect_type");
    if (!effectWidget) return;

    const activeParams = EFFECT_WIDGETS[effectWidget.value] || [];
    const visibleSet = new Set([...ALWAYS_VISIBLE, ...activeParams]);

    for (const w of node.widgets || []) {
        if (visibleSet.has(w.name)) {
            w.type = w._origType || w.type;
            delete w.computeSize;
        } else {
            if (!w._origType) w._origType = w.type;
            w.type = "hidden";
            w.computeSize = () => [0, -4];
        }
    }

    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true);
}

// ============================================================================
// CSS THEME — Orange + Graphite
// Target: .lg-node[data-node-id] is the Nodes 2.0 Vue container
// PrimeVue ToggleSwitch: .p-toggleswitch, checked = .p-toggleswitch-checked
// ============================================================================

// Track Koshi node IDs
const koshiNodeIds = new Set();

// Build CSS selectors for all registered Koshi node IDs
function buildKoshiCSS() {
    if (koshiNodeIds.size === 0) return "";

    const selectors = [...koshiNodeIds].map(id =>
        `.lg-node[data-node-id="${id}"]`
    );
    const sel = selectors.join(",\n");

    return `
/* === Koshi Nodes — Orange + Graphite === */

/* Node min-width */
${sel} {
    min-width: ${DEFAULT_WIDTH}px !important;
}

/* Override PrimeVue design tokens scoped to Koshi nodes */
:is(${sel}) {
    --p-toggleswitch-checked-background: #FF9F43;
    --p-toggleswitch-checked-hover-background: #e88e2e;
    --p-toggleswitch-background: #444;
    --p-toggleswitch-hover-background: #555;
    --p-toggleswitch-border-color: #555;
    --p-toggleswitch-checked-border-color: #FF9F43;
    --p-toggleswitch-handle-background: #1a1a1a;
    --p-toggleswitch-handle-checked-background: #1a1a1a;
    --p-slider-range-background: #FF9F43;
    --p-slider-handle-background: #FF9F43;
    --p-slider-handle-content-background: #FF9F43;
    --p-slider-track-background: #333;
    --p-slider-background: #333;
    --p-select-focus-border-color: #FF9F43;
    --p-focus-ring-color: rgba(255, 159, 67, 0.3);
    /* Number fader fill — bg-primary-background/15 uses color-mix(oklab, var(--primary-background) 15%) */
    --primary-background: #FF9F43;
    /* Slider range + thumb — bg-node-component-surface-highlight */
    --node-component-surface-highlight: #FF9F43;
}

/* Fallback direct overrides for older PrimeVue / non-token builds */
:is(${sel}) .p-toggleswitch-checked .p-toggleswitch-slider {
    background: #FF9F43 !important;
    border-color: #FF9F43 !important;
}
:is(${sel}) .p-toggleswitch .p-toggleswitch-slider {
    background: #444 !important;
    border-color: #555 !important;
}
:is(${sel}) .p-toggleswitch .p-toggleswitch-slider::before {
    background: #1a1a1a !important;
}

/* Native inputs */
:is(${sel}) input[type="checkbox"],
:is(${sel}) input[type="range"] {
    accent-color: #FF9F43 !important;
}

/* Number fader sliders — graphite track, orange fill */
:is(${sel}) .p-slider {
    background: #333 !important;
}
:is(${sel}) .p-slider .p-slider-range {
    background: #FF9F43 !important;
}
:is(${sel}) .p-slider .p-slider-handle {
    background: #FF9F43 !important;
    border-color: #FF9F43 !important;
}

/* Dropdown focus ring */
:is(${sel}) .p-select:focus-within,
:is(${sel}) .p-dropdown:focus-within {
    border-color: #FF9F43 !important;
    box-shadow: 0 0 0 1px rgba(255, 159, 67, 0.3) !important;
}
`;
}

// Inject and update the dynamic stylesheet
const koshiStyleEl = document.createElement("style");
koshiStyleEl.id = "koshi-nodes-theme";
document.head.appendChild(koshiStyleEl);

function refreshCSS() {
    koshiStyleEl.textContent = buildKoshiCSS();
}

// ============================================================================
// DIRECT DOM STYLING — MutationObserver for toggles + faders
// PrimeVue's scoped styles resist CSS overrides, so we set inline styles
// ============================================================================

function styleKoshiWidgets() {
    for (const nodeId of koshiNodeIds) {
        // Try multiple selectors — different ComfyUI versions use different DOM
        const el = document.querySelector(`[data-node-id="${nodeId}"]`)
            || document.querySelector(`.lg-node[data-node-id="${nodeId}"]`);
        if (!el) continue;

        // Style toggle switches — try all possible PrimeVue toggle selectors
        el.querySelectorAll("[class*='toggleswitch'], [class*='inputswitch'], [role='switch']").forEach(toggle => {
            const isChecked = toggle.classList.contains("p-toggleswitch-checked")
                || toggle.classList.contains("p-highlight")
                || toggle.getAttribute("data-p-checked") === "true"
                || toggle.getAttribute("data-p-highlight") === "true"
                || toggle.getAttribute("aria-checked") === "true";

            const bg = isChecked ? "#FF9F43" : "#444";
            const border = isChecked ? "#FF9F43" : "#555";

            // Set on the toggle root
            toggle.style.setProperty("--p-toggleswitch-checked-background", "#FF9F43");
            toggle.style.setProperty("--p-toggleswitch-background", "#444");

            // Find and style the slider/track and handle elements
            for (const child of toggle.querySelectorAll("*")) {
                const cls = child.className || "";
                if (typeof cls !== "string") continue;
                if (cls.includes("slider") || cls.includes("track")) {
                    child.style.setProperty("background", bg, "important");
                    child.style.setProperty("border-color", border, "important");
                    child.style.setProperty("transition", "background 0.15s", "important");
                }
                if (cls.includes("handle")) {
                    // Knob: orange when on, dark graphite when off
                    child.style.setProperty("background", isChecked ? "#fff" : "#222", "important");
                    child.style.setProperty("transition", "background 0.15s", "important");
                }
            }
        });

        // Style number fader fill bars (Tailwind bg-primary-background/15)
        // The fill div is inside: div.overflow-clip > div.bg-primary-background
        el.querySelectorAll("[class*='overflow-clip']").forEach(clip => {
            for (const child of clip.children) {
                const cls = child.className || "";
                if (typeof cls !== "string") continue;
                if (cls.includes("bg-primary-background") || cls.includes("size-full")) {
                    child.style.setProperty("background", "rgba(255, 159, 67, 0.15)", "important");
                }
            }
        });

        // Style slider range + thumb (Tailwind bg-node-component-surface-highlight)
        el.querySelectorAll("[data-slot='slider-range']").forEach(range => {
            range.style.setProperty("background", "#FF9F43", "important");
        });
        el.querySelectorAll("[data-slot='slider-thumb']").forEach(thumb => {
            thumb.style.setProperty("background", "#FF9F43", "important");
        });
        el.querySelectorAll("[data-slot='slider-track']").forEach(track => {
            track.style.setProperty("background", "#333", "important");
        });

        // PrimeVue slider fallback
        el.querySelectorAll(".p-slider:not(.p-toggleswitch-slider)").forEach(slider => {
            slider.style.setProperty("background", "#333", "important");
            for (const child of slider.querySelectorAll("*")) {
                const cls = child.className || "";
                if (typeof cls !== "string") continue;
                if (cls.includes("range")) {
                    child.style.setProperty("background", "#FF9F43", "important");
                }
                if (cls.includes("handle")) {
                    child.style.setProperty("background", "#FF9F43", "important");
                    child.style.setProperty("border-color", "#FF9F43", "important");
                }
            }
        });
    }
}

// Debug: log what we find on first run
setTimeout(() => {
    for (const nodeId of koshiNodeIds) {
        const el = document.querySelector(`[data-node-id="${nodeId}"]`);
        if (el) {
            const toggles = el.querySelectorAll("[class*='toggleswitch'], [class*='inputswitch'], [role='switch']");
            console.log(`[Koshi] Node ${nodeId}: found ${toggles.length} toggles`, el.className);
            if (toggles.length > 0) console.log("[Koshi] Toggle classes:", toggles[0].className);
        } else {
            console.log(`[Koshi] Node ${nodeId}: DOM element NOT FOUND`);
        }
    }
}, 2000);

// Observe DOM changes to restyle after Vue renders — throttled
let styleTimeout = null;
const widgetObserver = new MutationObserver(() => {
    if (koshiNodeIds.size === 0) return;
    if (styleTimeout) return;
    styleTimeout = setTimeout(() => {
        styleKoshiWidgets();
        styleTimeout = null;
    }, 50);
});
setTimeout(() => {
    widgetObserver.observe(document.body, {
        childList: true, subtree: true, attributes: true,
        attributeFilter: ["class", "aria-checked", "data-p-checked", "data-p-highlight"]
    });
    styleKoshiWidgets();
}, 1000);

// ============================================================================
// EXTENSION
// ============================================================================

app.registerExtension({
    name: "Koshi-Nodes.appearance",
    nodeCreated(node) {
        const category = NODE_CATEGORIES[node.comfyClass];
        if (category && COLORS[category]) {
            node.color = COLORS[category].color;
            node.bgcolor = COLORS[category].bgcolor;

            // Register node ID for CSS targeting
            // node.id is -1 at creation time, capture real ID after assignment
            const registerKoshiId = () => {
                koshiNodeIds.delete("-1");
                koshiNodeIds.add(String(node.id));
                refreshCSS();
                styleKoshiWidgets();
            };
            // Hook into onAdded (fires when node gets real ID in graph)
            const origAdded = node.onAdded;
            node.onAdded = function(graph) {
                if (origAdded) origAdded.call(this, graph);
                registerKoshiId();
            };
            // Also try delayed (fallback for loaded workflows)
            setTimeout(registerKoshiId, 300);
            setTimeout(registerKoshiId, 1000);

            // Enforce minimum width — hook into onConfigure (fires after load)
            // and use direct size mutation + delayed recompute
            node.size[0] = Math.max(node.size[0], DEFAULT_WIDTH);

            const origConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                this.size[0] = Math.max(this.size[0], DEFAULT_WIDTH);
            };

            // Also after all widgets are added (ComfyUI recomputes size)
            const origCompute = node.computeSize;
            node.computeSize = function() {
                const sz = origCompute ? origCompute.call(this) : [DEFAULT_WIDTH, 200];
                sz[0] = Math.max(sz[0], DEFAULT_WIDTH);
                return sz;
            };
        }

        // Dynamic widget visibility for Koshi Effects
        if (node.comfyClass === "Koshi_Effects") {
            node.size = [DEFAULT_WIDTH, 400];

            const origChanged = node.onWidgetChanged;
            node.onWidgetChanged = function(name, value, old_value, widget) {
                if (origChanged) origChanged.call(this, name, value, old_value, widget);
                if (name === "effect_type") {
                    updateEffectWidgets(this);
                }
            };
            setTimeout(() => updateEffectWidgets(node), 100);
        }
    }
});
