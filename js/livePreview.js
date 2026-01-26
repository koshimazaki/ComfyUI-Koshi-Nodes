import { app } from "../../scripts/app.js";

/**
 * Koshi Nodes Live Preview Extension
 * Adds inline image preview to generator, effect, and OLED nodes.
 * Includes interactive orbital controls for 3D raymarched generators.
 */

const PREVIEW_NODES = [
    // Generators
    "Koshi_GlitchCandies",
    "Koshi_ShapeMorph",
    "Koshi_NoiseDisplace",
    // Effects
    "Koshi_Hologram",
    "Koshi_Scanlines",
    "Koshi_VideoGlitch",
    "Koshi_ChromaticAberration",
    "Koshi_Bloom",
    "KoshiBloomShader",
    "Koshi_Glitch",
    "Koshi_Raymarcher",
    // OLED/Export
    "Koshi_OLEDScreen",
    "Koshi_PixelScaler",
    "Koshi_SpriteSheet",
    // SIDKIT Image
    "Koshi_Dither",
    "Koshi_Binary",
    "Koshi_Greyscale",
];

// 3D nodes that get orbital controls
const ORBITAL_NODES = ["Koshi_GlitchCandies"];

// 3D patterns that support rotation
const PATTERNS_3D = ["rm_cube", "rm_sphere", "rm_torus", "rm_octahedron", "rm_gyroid", "rm_menger"];

app.registerExtension({
    name: "Koshi.LivePreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!PREVIEW_NODES.includes(nodeData.name)) {
            return;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);

            if (message?.images) {
                this.showPreview(message.images);
            }
        };

        nodeType.prototype.showPreview = function(images) {
            if (!images || images.length === 0) return;

            // Check if preview is disabled
            const toggleWidget = this.widgets?.find(w => w.name === "show_preview");
            if (toggleWidget && !toggleWidget.value) return;

            // Get or create preview widget
            let previewWidget = this.widgets?.find(w => w.name === "koshi_preview");

            if (!previewWidget) {
                previewWidget = this.addDOMWidget("koshi_preview", "preview", document.createElement("div"), {
                    serialize: false,
                    hideOnZoom: false,
                });
                previewWidget.computeSize = () => [this.size[0], 220];
            }

            const container = previewWidget.element;
            container.innerHTML = "";
            container.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                padding: 8px;
                background: #1a1a1a;
                border-radius: 4px;
                overflow: hidden;
            `;

            // Image container
            const imgContainer = document.createElement("div");
            imgContainer.style.cssText = `
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                justify-content: center;
            `;

            // Calculate image size based on node width
            const nodeWidth = this.size[0] - 32;
            const imgCount = Math.min(images.length, 4);
            const imgSize = Math.min(180, Math.floor(nodeWidth / Math.min(imgCount, 2)) - 8);

            images.slice(0, 4).forEach((imgData, idx) => {
                const img = document.createElement("img");
                img.style.cssText = `
                    max-width: ${imgSize}px;
                    max-height: ${imgSize}px;
                    object-fit: contain;
                    border: 1px solid #333;
                    border-radius: 2px;
                    image-rendering: pixelated;
                `;

                // Build image URL
                const params = new URLSearchParams({
                    filename: imgData.filename,
                    subfolder: imgData.subfolder || "",
                    type: imgData.type || "temp",
                });
                img.src = `/view?${params.toString()}`;
                img.title = `Frame ${idx + 1}`;

                imgContainer.appendChild(img);
            });

            container.appendChild(imgContainer);

            // Show frame count if more than 4
            if (images.length > 4) {
                const more = document.createElement("div");
                more.style.cssText = `
                    color: #888;
                    font-size: 11px;
                    width: 100%;
                    text-align: center;
                `;
                more.textContent = `+${images.length - 4} more frames`;
                container.appendChild(more);
            }

            // Add orbital controls for 3D generators
            if (ORBITAL_NODES.includes(this.comfyClass)) {
                this.addOrbitalControls(container, imgContainer.querySelector("img"));
            }

            // Resize node to fit preview
            const minHeight = 250 + this.computeSize()[1];
            if (this.size[1] < minHeight) {
                this.setSize([this.size[0], minHeight]);
            }

            app.graph.setDirtyCanvas(true);
        };

        // Add orbital controls for 3D nodes
        nodeType.prototype.addOrbitalControls = function(container, previewImg) {
            // Check if current pattern is 3D
            const patternWidget = this.widgets?.find(w => w.name === "pattern");
            if (!patternWidget || !PATTERNS_3D.includes(patternWidget.value)) {
                return;
            }

            const rotXWidget = this.widgets?.find(w => w.name === "rotation_x");
            const rotYWidget = this.widgets?.find(w => w.name === "rotation_y");
            const camDistWidget = this.widgets?.find(w => w.name === "camera_distance");

            if (!rotXWidget || !rotYWidget) return;

            // Create controls container
            const controls = document.createElement("div");
            controls.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                margin-top: 8px;
                padding: 8px;
                background: #252525;
                border-radius: 4px;
            `;

            // Title
            const title = document.createElement("div");
            title.style.cssText = "color: #03d7fc; font-size: 11px; font-weight: bold; margin-bottom: 4px;";
            title.textContent = "3D Controls (drag image to rotate)";
            controls.appendChild(title);

            // Make preview draggable for rotation
            if (previewImg) {
                let isDragging = false;
                let lastX = 0, lastY = 0;

                previewImg.style.cursor = "grab";

                previewImg.addEventListener("mousedown", (e) => {
                    isDragging = true;
                    lastX = e.clientX;
                    lastY = e.clientY;
                    previewImg.style.cursor = "grabbing";
                    e.preventDefault();
                });

                document.addEventListener("mousemove", (e) => {
                    if (!isDragging) return;

                    const deltaX = e.clientX - lastX;
                    const deltaY = e.clientY - lastY;

                    // Update rotation widgets
                    rotYWidget.value = ((rotYWidget.value || 0) + deltaX * 0.5) % 360;
                    rotXWidget.value = Math.max(-90, Math.min(90, (rotXWidget.value || 0) + deltaY * 0.5));

                    // Update widget UI
                    if (rotYWidget.callback) rotYWidget.callback(rotYWidget.value);
                    if (rotXWidget.callback) rotXWidget.callback(rotXWidget.value);

                    lastX = e.clientX;
                    lastY = e.clientY;

                    app.graph.setDirtyCanvas(true);
                });

                document.addEventListener("mouseup", () => {
                    if (isDragging) {
                        isDragging = false;
                        previewImg.style.cursor = "grab";
                        // Queue node for re-execution
                        if (this.graph) {
                            app.queuePrompt();
                        }
                    }
                });

                // Mouse wheel for zoom
                previewImg.addEventListener("wheel", (e) => {
                    if (camDistWidget) {
                        const delta = e.deltaY > 0 ? 0.2 : -0.2;
                        camDistWidget.value = Math.max(1, Math.min(10, (camDistWidget.value || 3) + delta));
                        if (camDistWidget.callback) camDistWidget.callback(camDistWidget.value);
                        app.queuePrompt();
                    }
                    e.preventDefault();
                }, { passive: false });
            }

            // Rotation display
            const rotDisplay = document.createElement("div");
            rotDisplay.style.cssText = "color: #aaa; font-size: 10px; font-family: monospace;";
            rotDisplay.textContent = `Rotation: X=${(rotXWidget.value || 0).toFixed(1)}° Y=${(rotYWidget.value || 0).toFixed(1)}°`;
            controls.appendChild(rotDisplay);

            // Reset button
            const resetBtn = document.createElement("button");
            resetBtn.textContent = "Reset View";
            resetBtn.style.cssText = `
                background: #333;
                border: 1px solid #555;
                color: #fff;
                padding: 4px 8px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                margin-top: 4px;
            `;
            resetBtn.addEventListener("click", () => {
                rotXWidget.value = 0;
                rotYWidget.value = 0;
                if (camDistWidget) camDistWidget.value = 3;
                if (rotXWidget.callback) rotXWidget.callback(0);
                if (rotYWidget.callback) rotYWidget.callback(0);
                if (camDistWidget?.callback) camDistWidget.callback(3);
                rotDisplay.textContent = "Rotation: X=0.0° Y=0.0°";
                app.queuePrompt();
            });
            controls.appendChild(resetBtn);

            // Update display on widget change
            const updateDisplay = () => {
                rotDisplay.textContent = `Rotation: X=${(rotXWidget.value || 0).toFixed(1)}° Y=${(rotYWidget.value || 0).toFixed(1)}°`;
            };

            // Hook widget callbacks
            const origRotXCb = rotXWidget.callback;
            rotXWidget.callback = (v) => { origRotXCb?.(v); updateDisplay(); };
            const origRotYCb = rotYWidget.callback;
            rotYWidget.callback = (v) => { origRotYCb?.(v); updateDisplay(); };

            container.appendChild(controls);
        };
    },

    async nodeCreated(node) {
        if (!PREVIEW_NODES.includes(node.comfyClass)) {
            return;
        }

        // Add preview toggle
        const showPreview = node.addWidget("toggle", "show_preview", true, () => {
            const previewWidget = node.widgets?.find(w => w.name === "koshi_preview");
            if (previewWidget) {
                previewWidget.element.style.display = showPreview.value ? "flex" : "none";
                node.setDirtyCanvas(true);
            }
        });
        showPreview.serialize = false;
    },
});
