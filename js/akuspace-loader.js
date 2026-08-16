/**
 * Lazy loader for the AKUSPACE spatial widget
 *
 * The widget bundle carries a Vue runtime and Three.js (~980KB). ComfyUI
 * auto-loads every `**\/*.js` under WEB_DIRECTORY, so the bundle lives at
 * `akuspace-widget.mjs` to stay out of that glob and is imported here only
 * once an AKUSPACE node actually exists. Same approach ComfyUI core takes for
 * Load3D/Three.js in `extensions/core/load3dLazy.ts`.
 */

import { app } from "../../../scripts/app.js";

const WIDGET_URL = new URL("./akuspace-widget.mjs", import.meta.url).href;
const WIDGET_EXTENSION = "Koshi.AKUSPACE.SpatialControl";
const AKUSPACE_NODES = new Set([
    "Koshi_AKUSPACEPrompt",
    "Koshi_AKUSPACETextEncode",
]);

let widgetPromise = null;

function nodeTypeName(node) {
    return (
        node?.constructor?.comfyClass ??
        node?.comfyClass ??
        node?.type ??
        node?.constructor?.type
    );
}

function loadWidget() {
    if (!widgetPromise) {
        widgetPromise = import(/* webpackIgnore: true */ WIDGET_URL).catch((error) => {
            widgetPromise = null; // let a later node retry
            console.error("[Koshi] AKUSPACE widget failed to load:", error);
            throw error;
        });
    }
    return widgetPromise;
}

app.registerExtension({
    name: "Koshi.AKUSPACE.Loader",

    async nodeCreated(node) {
        if (!AKUSPACE_NODES.has(nodeTypeName(node))) return;

        const alreadyLoaded = app.extensions?.some((ext) => ext.name === WIDGET_EXTENSION);
        if (alreadyLoaded) return; // the widget's own hook covers this node

        try {
            await loadWidget();
        } catch {
            return; // already reported above; leave the node usable without the widget
        }

        // `invokeExtensionsAsync` snapshots the extension list before this
        // await, so the widget missed its own nodeCreated for this node.
        const widget = app.extensions?.find((ext) => ext.name === WIDGET_EXTENSION);
        widget?.nodeCreated?.(node, app);
    },
});
