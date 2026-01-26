/**
 * Koshi Live Preview System
 * Real-time WebGL preview for Koshi effects nodes
 * Includes OLED screen emulation for SIDKIT export preview
 */

import { app } from "../../../scripts/app.js";

// ============================================================================
// GLSL SHADERS
// ============================================================================

const VERTEX_SHADER = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
    }
`;

const PASSTHROUGH_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    varying vec2 v_texCoord;
    void main() {
        gl_FragColor = texture2D(u_image, v_texCoord);
    }
`;

// Bayer 8x8 Dithering
const DITHER_BAYER_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_intensity;
    uniform float u_levels;
    uniform float u_pixel_size;
    varying vec2 v_texCoord;
    
    float bayer8(vec2 pos) {
        vec2 p = mod(pos, 8.0);
        float idx = p.x + p.y * 8.0;
        // Bayer 8x8 pattern approximation
        float b = mod(p.x + p.y * 2.0, 4.0) + mod(p.x * 2.0 + p.y, 4.0) * 4.0;
        return b / 16.0;
    }
    
    void main() {
        vec2 uv = v_texCoord;
        
        // Pixelation
        if (u_pixel_size > 1.0) {
            vec2 pixelSize = u_pixel_size / u_resolution;
            uv = floor(uv / pixelSize) * pixelSize;
        }
        
        vec4 color = texture2D(u_image, uv);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        
        vec2 pixelPos = v_texCoord * u_resolution / max(u_pixel_size, 1.0);
        float threshold = bayer8(pixelPos);
        
        float steps = max(1.0, u_levels - 1.0);
        float dithered = gray + (threshold - 0.5) * u_intensity / steps;
        dithered = floor(dithered * steps + 0.5) / steps;
        
        gl_FragColor = vec4(vec3(clamp(dithered, 0.0, 1.0)), color.a);
    }
`;

// Floyd-Steinberg approximation
const DITHER_FS_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_levels;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        
        // Simple noise-based FS approximation
        vec2 p = v_texCoord * u_resolution;
        float noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
        
        float steps = max(1.0, u_levels - 1.0);
        float quantized = floor(gray * steps + noise * 0.5) / steps;
        
        gl_FragColor = vec4(vec3(clamp(quantized, 0.0, 1.0)), color.a);
    }
`;

// Halftone dithering
const DITHER_HALFTONE_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_dot_size;
    uniform float u_angle;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        
        float angle = u_angle * 3.14159 / 180.0;
        vec2 p = v_texCoord * u_resolution;
        
        // Rotate
        vec2 rotated = vec2(
            p.x * cos(angle) - p.y * sin(angle),
            p.x * sin(angle) + p.y * cos(angle)
        );
        
        vec2 cell = mod(rotated, u_dot_size) - u_dot_size * 0.5;
        float dist = length(cell) / (u_dot_size * 0.5);
        
        float dot = step(dist, 1.0 - gray);
        
        gl_FragColor = vec4(vec3(dot), color.a);
    }
`;

// Glitch shader
const GLITCH_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_glitch_intensity;
    uniform float u_rgb_shift;
    uniform float u_time;
    varying vec2 v_texCoord;
    
    float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
    }
    
    void main() {
        vec2 uv = v_texCoord;
        float intensity = u_glitch_intensity;
        
        // RGB shift
        float shift = u_rgb_shift / u_resolution.x * intensity;
        float r = texture2D(u_image, uv + vec2(shift, 0.0)).r;
        float g = texture2D(u_image, uv).g;
        float b = texture2D(u_image, uv - vec2(shift, 0.0)).b;
        
        // Block glitch
        float blockY = floor(uv.y * 20.0);
        float noise = rand(vec2(blockY, u_time));
        if (noise > 1.0 - intensity * 0.3) {
            float blockShift = (rand(vec2(blockY, u_time + 1.0)) - 0.5) * 0.1 * intensity;
            r = texture2D(u_image, uv + vec2(blockShift + shift, 0.0)).r;
            g = texture2D(u_image, uv + vec2(blockShift, 0.0)).g;
            b = texture2D(u_image, uv + vec2(blockShift - shift, 0.0)).b;
        }
        
        gl_FragColor = vec4(r, g, b, 1.0);
    }
`;

// Bloom shader
const BLOOM_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_threshold;
    uniform float u_intensity;
    uniform float u_radius;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        vec3 bloom = vec3(0.0);
        
        vec2 texelSize = 1.0 / u_resolution;
        float samples = 0.0;
        
        for (float x = -4.0; x <= 4.0; x += 1.0) {
            for (float y = -4.0; y <= 4.0; y += 1.0) {
                vec2 offset = vec2(x, y) * texelSize * u_radius * 8.0;
                vec4 s = texture2D(u_image, v_texCoord + offset);
                float bright = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
                if (bright > u_threshold) {
                    bloom += s.rgb * (bright - u_threshold);
                    samples += 1.0;
                }
            }
        }
        
        if (samples > 0.0) bloom /= samples;
        
        gl_FragColor = vec4(color.rgb + bloom * u_intensity, color.a);
    }
`;

// Threshold (binary)
const THRESHOLD_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform float u_threshold;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        float binary = step(u_threshold, gray);
        gl_FragColor = vec4(vec3(binary), color.a);
    }
`;

// Greyscale (luminosity)
const GREYSCALE_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform float u_amount;
    varying vec2 v_texCoord;
    
    void main() {
        vec4 color = texture2D(u_image, v_texCoord);
        float gray = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        vec3 result = mix(color.rgb, vec3(gray), u_amount);
        gl_FragColor = vec4(result, color.a);
    }
`;

// OLED Screen Emulation (256x128, 4-bit greyscale with pixel grid)
const OLED_EMULATION_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform vec2 u_screenSize;  // 256x128
    uniform float u_bitDepth;   // 4 = 16 levels
    uniform float u_pixelGap;
    uniform float u_brightness;
    uniform float u_contrast;
    varying vec2 v_texCoord;
    
    void main() {
        // Map to screen pixels
        vec2 screenUV = v_texCoord;
        vec2 pixelCoord = floor(screenUV * u_screenSize);
        vec2 pixelUV = pixelCoord / u_screenSize;
        
        // Sample and convert to greyscale
        vec4 color = texture2D(u_image, pixelUV + 0.5 / u_screenSize);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        
        // Apply brightness/contrast
        gray = (gray - 0.5) * u_contrast + 0.5 + u_brightness;
        gray = clamp(gray, 0.0, 1.0);
        
        // Quantize to bit depth (4-bit = 16 levels)
        float levels = pow(2.0, u_bitDepth);
        gray = floor(gray * (levels - 1.0) + 0.5) / (levels - 1.0);
        
        // OLED pixel effect - pixels are slightly separated
        vec2 pixelPos = fract(screenUV * u_screenSize);
        float pixelMask = 1.0;
        if (u_pixelGap > 0.0) {
            float gap = u_pixelGap;
            if (pixelPos.x < gap || pixelPos.x > 1.0 - gap ||
                pixelPos.y < gap || pixelPos.y > 1.0 - gap) {
                pixelMask = 0.1;  // Dark gap between pixels
            }
        }
        
        // OLED: true black when off, slight blue tint on bright
        vec3 oledColor;
        if (gray < 0.01) {
            oledColor = vec3(0.0);  // True black
        } else {
            // Slight cool tint for OLED look
            oledColor = vec3(gray * 0.95, gray, gray * 1.02);
        }
        
        gl_FragColor = vec4(oledColor * pixelMask, 1.0);
    }
`;

// ============================================================================
// PREVIEW ENGINE
// ============================================================================

class KoshiPreviewEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl", { preserveDrawingBuffer: true });
        if (!this.gl) {
            console.error("[Koshi] WebGL not available");
            return;
        }
        
        this.programs = {};
        this.sourceTexture = null;
        this.imageLoaded = false;
        
        this._initGeometry();
        this._initPrograms();
    }
    
    _initGeometry() {
        const gl = this.gl;
        
        const positions = new Float32Array([
            -1, -1,  1, -1,  -1, 1,
            -1,  1,  1, -1,   1, 1
        ]);
        
        const texCoords = new Float32Array([
            0, 0,  1, 0,  0, 1,
            0, 1,  1, 0,  1, 1
        ]);
        
        this.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        
        this.texCoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    }
    
    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error("[Koshi] Shader error:", gl.getShaderInfoLog(shader));
            return null;
        }
        return shader;
    }
    
    _createProgram(fragmentSource) {
        const gl = this.gl;
        const vertexShader = this._compileShader(gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragmentShader = this._compileShader(gl.FRAGMENT_SHADER, fragmentSource);
        if (!vertexShader || !fragmentShader) return null;
        
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error("[Koshi] Program link error:", gl.getProgramInfoLog(program));
            return null;
        }
        return program;
    }
    
    _initPrograms() {
        this.programs.passthrough = this._createProgram(PASSTHROUGH_SHADER);
        this.programs.dither_bayer = this._createProgram(DITHER_BAYER_SHADER);
        this.programs.dither_fs = this._createProgram(DITHER_FS_SHADER);
        this.programs.dither_halftone = this._createProgram(DITHER_HALFTONE_SHADER);
        this.programs.glitch = this._createProgram(GLITCH_SHADER);
        this.programs.bloom = this._createProgram(BLOOM_SHADER);
        this.programs.threshold = this._createProgram(THRESHOLD_SHADER);
        this.programs.greyscale = this._createProgram(GREYSCALE_SHADER);
        this.programs.oled = this._createProgram(OLED_EMULATION_SHADER);
    }
    
    loadImage(imageElement) {
        const gl = this.gl;
        
        if (this.sourceTexture) {
            gl.deleteTexture(this.sourceTexture);
        }
        
        this.sourceTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageElement);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        
        this.imageWidth = imageElement.width || imageElement.naturalWidth;
        this.imageHeight = imageElement.height || imageElement.naturalHeight;
        this.imageLoaded = true;
    }
    
    render(effectType, params) {
        const gl = this.gl;
        if (!this.imageLoaded || !this.sourceTexture) return;
        
        const program = this.programs[effectType] || this.programs.passthrough;
        if (!program) return;
        
        gl.useProgram(program);
        
        // Set up attributes
        const posLoc = gl.getAttribLocation(program, "a_position");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        
        const texLoc = gl.getAttribLocation(program, "a_texCoord");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.enableVertexAttribArray(texLoc);
        gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
        
        // Set uniforms
        const imageLoc = gl.getUniformLocation(program, "u_image");
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.sourceTexture);
        gl.uniform1i(imageLoc, 0);
        
        const resLoc = gl.getUniformLocation(program, "u_resolution");
        if (resLoc) gl.uniform2f(resLoc, this.canvas.width, this.canvas.height);
        
        // Effect-specific uniforms
        for (const [key, value] of Object.entries(params || {})) {
            const loc = gl.getUniformLocation(program, `u_${key}`);
            if (loc !== null) {
                if (Array.isArray(value)) {
                    if (value.length === 2) gl.uniform2f(loc, value[0], value[1]);
                } else {
                    gl.uniform1f(loc, value);
                }
            }
        }
        
        // Render
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}

// ============================================================================
// PREVIEW WIDGET
// ============================================================================

class KoshiPreviewWidget {
    constructor(node, effectType, options = {}) {
        this.node = node;
        this.effectType = effectType;
        this.options = options;
        this.engine = null;
        this.previewImage = null;
        this.params = {};
        this.visible = true;
        this.animationFrame = null;
        this.time = 0;
        
        this._createUI();
    }
    
    _createUI() {
        this.container = document.createElement("div");
        this.container.className = "koshi-preview-container";
        this.container.style.cssText = `
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
            border-radius: 4px;
            overflow: hidden;
            margin: 4px;
        `;
        
        // Header with toggle
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 8px;
            background: #252525;
            cursor: pointer;
        `;
        
        const title = document.createElement("span");
        title.style.cssText = "font-size: 10px; color: #aaa;";
        title.textContent = this.options.title || "Preview";
        header.appendChild(title);
        
        const toggle = document.createElement("span");
        toggle.style.cssText = "font-size: 10px; color: #666;";
        toggle.textContent = "▼";
        header.appendChild(toggle);
        
        header.onclick = () => {
            this.visible = !this.visible;
            this.previewWrapper.style.display = this.visible ? "block" : "none";
            toggle.textContent = this.visible ? "▼" : "▶";
            this.node.setSize([this.node.size[0], this.node.computeSize()[1]]);
            this.node.graph?.setDirtyCanvas(true);
        };
        
        this.container.appendChild(header);
        
        // Preview wrapper
        this.previewWrapper = document.createElement("div");
        this.previewWrapper.style.cssText = "position: relative;";
        
        // Canvas for preview
        const canvasWidth = this.options.width || 256;
        const canvasHeight = this.options.height || 128;
        
        this.canvas = document.createElement("canvas");
        this.canvas.width = canvasWidth;
        this.canvas.height = canvasHeight;
        this.canvas.style.cssText = `
            width: 100%;
            height: auto;
            background: #000;
            image-rendering: pixelated;
        `;
        this.previewWrapper.appendChild(this.canvas);
        
        // OLED bezel effect (optional)
        if (this.options.oledBezel) {
            this.canvas.style.cssText += `
                border: 3px solid #111;
                border-radius: 2px;
                box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
            `;
        }
        
        this.container.appendChild(this.previewWrapper);
        
        // Status text
        this.status = document.createElement("div");
        this.status.style.cssText = `
            padding: 4px 8px;
            font-size: 10px;
            color: #666;
            text-align: center;
        `;
        this.status.textContent = this.options.statusText || "Connect image to preview";
        this.container.appendChild(this.status);
        
        // Initialize WebGL engine
        this.engine = new KoshiPreviewEngine(this.canvas);
    }
    
    getElement() {
        return this.container;
    }
    
    setImage(imageUrl) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this.engine.loadImage(img);
            this.previewImage = img;
            this.status.textContent = `Source: ${img.width}×${img.height}`;
            this.render();
        };
        img.onerror = () => {
            this.status.textContent = "Failed to load image";
        };
        img.src = imageUrl;
    }
    
    setParams(params) {
        this.params = { ...this.params, ...params };
        this.render();
    }
    
    render() {
        if (this.engine && this.previewImage) {
            this.engine.render(this.effectType, this.params);
        }
    }
    
    startAnimation() {
        if (this.animationFrame) return;
        
        const animate = () => {
            this.time += 0.1;
            this.params.time = this.time;
            this.render();
            this.animationFrame = requestAnimationFrame(animate);
        };
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    getHeight() {
        if (!this.visible) return 30;
        return (this.options.height || 128) + 60;
    }
}

// ============================================================================
// COMFYUI EXTENSION
// ============================================================================

app.registerExtension({
    name: "Koshi.LivePreview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // Map nodes to their preview configuration
        const previewConfigs = {
            // Dithering nodes
            "Koshi_DitheringFilter": { 
                effect: "dither_bayer", 
                title: "Dither Preview",
                paramMap: { 
                    dither_intensity: "intensity", 
                    color_steps: "levels",
                    pixel_size: "pixel_size"
                }
            },
            "Koshi_FloydSteinberg": { 
                effect: "dither_fs", 
                title: "Floyd-Steinberg Preview",
                paramMap: { levels: "levels" }
            },
            "Koshi_Bayer": { 
                effect: "dither_bayer", 
                title: "Bayer Preview",
                paramMap: { levels: "levels" }
            },
            "Koshi_Halftone": { 
                effect: "dither_halftone", 
                title: "Halftone Preview",
                paramMap: { dot_size: "dot_size", angle: "angle" }
            },
            
            // Greyscale nodes
            "Koshi_Luminosity": { 
                effect: "greyscale", 
                title: "Greyscale Preview",
                paramMap: {},
                defaultParams: { amount: 1.0 }
            },
            "Koshi_Desaturate": { 
                effect: "greyscale", 
                title: "Desaturate Preview",
                paramMap: { amount: "amount" }
            },
            "Koshi_ChannelMono": { 
                effect: "greyscale", 
                title: "Channel Preview",
                paramMap: {},
                defaultParams: { amount: 1.0 }
            },
            
            // Binary nodes
            "Koshi_Threshold": { 
                effect: "threshold", 
                title: "Threshold Preview",
                paramMap: { threshold: "threshold" }
            },
            
            // Effects nodes
            "Koshi_Glitch": { 
                effect: "glitch", 
                title: "Glitch Preview",
                animated: true,
                paramMap: { 
                    glitch_intensity: "glitch_intensity",
                    rgb_shift: "rgb_shift"
                }
            },
            "Koshi_Bloom": { 
                effect: "bloom", 
                title: "Bloom Preview",
                paramMap: { 
                    threshold: "threshold",
                    intensity: "intensity",
                    radius: "radius"
                }
            },
        };
        
        const config = previewConfigs[nodeData.name];
        if (!config) return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            
            const widget = new KoshiPreviewWidget(this, config.effect, {
                title: config.title,
                width: 256,
                height: 192,
            });
            
            // Set default params
            if (config.defaultParams) {
                widget.setParams(config.defaultParams);
            }
            
            const domWidget = this.addDOMWidget("koshi_preview", "preview", widget.getElement(), {
                serialize: false,
                hideOnZoom: false,
            });
            
            domWidget.computeSize = (width) => [width, widget.getHeight()];
            this.koshiPreview = widget;
            this.koshiPreviewConfig = config;
            
            // Start animation for animated effects
            if (config.animated) {
                widget.startAnimation();
            }
        };
        
        // Update preview when widgets change
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function(name, value) {
            if (onPropertyChanged) onPropertyChanged.apply(this, arguments);
            this._updateKoshiPreview?.();
        };
        
        // Widget callback for slider changes
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(name, value, old_value, widget) {
            if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            this._updateKoshiPreview?.();
        };
        
        nodeType.prototype._updateKoshiPreview = function() {
            if (!this.koshiPreview || !this.koshiPreviewConfig) return;
            
            const params = {};
            const paramMap = this.koshiPreviewConfig.paramMap || {};
            
            for (const w of this.widgets || []) {
                if (w.name === "koshi_preview") continue;
                const mappedName = paramMap[w.name] || w.name;
                params[mappedName] = w.value;
            }
            
            this.koshiPreview.setParams(params);
        };
        
        // Handle execution result
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            
            if (this.koshiPreview && message?.images?.length > 0) {
                const img = message.images[0];
                const url = `/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder || "")}`;
                this.koshiPreview.setImage(url);
            }
        };
    }
});

// ============================================================================
// OLED SCREEN PREVIEW NODE
// ============================================================================

app.registerExtension({
    name: "Koshi.OLEDPreview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Koshi_SIDKITExport" && nodeData.name !== "Koshi_OLEDPreview") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            
            const widget = new KoshiPreviewWidget(this, "oled", {
                title: "OLED 256×128 Preview",
                width: 256,
                height: 128,
                oledBezel: true,
                statusText: "SSD1363 4-bit OLED Emulation",
            });
            
            // Default OLED params
            widget.setParams({
                screenSize: [256, 128],
                bitDepth: 4,
                pixelGap: 0.05,
                brightness: 0.0,
                contrast: 1.0,
            });
            
            const domWidget = this.addDOMWidget("oled_preview", "preview", widget.getElement(), {
                serialize: false,
                hideOnZoom: false,
            });
            
            domWidget.computeSize = (width) => [width, widget.getHeight() + 20];
            this.oledPreview = widget;
        };
        
        // Update OLED preview with params
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(name, value) {
            if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            
            if (this.oledPreview) {
                const params = {
                    screenSize: [256, 128],
                    bitDepth: 4,
                    pixelGap: 0.05,
                    brightness: 0.0,
                    contrast: 1.0,
                };
                
                // Map widget values
                for (const w of this.widgets || []) {
                    if (w.name === "target_width") params.screenSize[0] = w.value;
                    if (w.name === "target_height") params.screenSize[1] = w.value;
                    if (w.name === "bit_depth") {
                        if (w.value.includes("1-bit")) params.bitDepth = 1;
                        else if (w.value.includes("2-bit")) params.bitDepth = 2;
                        else if (w.value.includes("4-bit")) params.bitDepth = 4;
                    }
                }
                
                this.oledPreview.setParams(params);
            }
        };
        
        // Handle execution result
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            
            if (this.oledPreview && message?.images?.length > 0) {
                const img = message.images[0];
                const url = `/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder || "")}`;
                this.oledPreview.setImage(url);
            }
        };
    }
});

console.log("[Koshi] Live Preview + OLED Emulation loaded");
