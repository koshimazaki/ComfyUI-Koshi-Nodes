/**
 * Koshi Glitch Candies - Live WebGL Preview
 * Real-time procedural pattern generator with ThreeJS-style continuous animation
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// GLSL Shader Library
const GLSL_COMMON = `
precision highp float;

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_scale;
uniform int u_seed;
uniform float u_camera_distance;
uniform float u_rotation_x;
uniform float u_rotation_y;

// Hash functions
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash3(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

// Value noise
float valueNoise(vec2 p, float t) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i) + 0.2 * sin(t + hash(i) * 6.28318);
    float b = hash(i + vec2(1.0, 0.0)) + 0.2 * sin(t + hash(i + vec2(1.0, 0.0)) * 6.28318);
    float c = hash(i + vec2(0.0, 1.0)) + 0.2 * sin(t + hash(i + vec2(0.0, 1.0)) * 6.28318);
    float d = hash(i + vec2(1.0, 1.0)) + 0.2 * sin(t + hash(i + vec2(1.0, 1.0)) * 6.28318);

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// FBM
float fbm(vec2 p, float t) {
    float result = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        result += amp * valueNoise(p, t + float(i) * 0.5);
        p *= 2.0;
        amp *= 0.5;
    }
    return result;
}

// Rotation matrices
mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0, 0,c,-s, 0,s,c);
}

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s, 0,1,0, -s,0,c);
}
`;

const GLSL_PATTERNS = {
    // 2D Patterns
    waves: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float t = u_time;

    float result = sin(uv.x * 20.0 + t) * sin(uv.y * 20.0 + t * 0.7);
    result += sin(length(uv - 0.5) * 30.0 - t * 2.0) * 0.5;
    result = result * 0.5 + 0.5;

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    circles: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float d = length(uv - 0.5);
    float result = sin(d * 40.0 - u_time * 3.0) * 0.5 + 0.5;
    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    plasma: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float t = u_time;

    float v1 = sin(uv.x * 10.0 + t);
    float v2 = sin(uv.y * 10.0 + t * 1.2);
    float v3 = sin((uv.x + uv.y) * 10.0 + t * 0.8);
    float v4 = sin(length(uv - 0.5) * 20.0 + t * 1.5);
    float result = (v1 + v2 + v3 + v4) * 0.25 + 0.5;

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    voronoi: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float loopTime = mod(u_time, 10.0) * 0.628318;
    vec2 p = uv * 5.0 * u_scale;
    vec2 ip = floor(p);
    vec2 fp = fract(p);

    float minD = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 cellId = ip + g;
            float h = hash(cellId);
            float o = 0.5 + 0.4 * sin(loopTime + h * 6.28318);
            vec2 r = g + o - fp;
            float d = length(r);
            minD = min(minD, d);
        }
    }

    gl_FragColor = vec4(vec3(minD), 1.0);
}`,

    glitch_candies: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float loopTime = mod(u_time, 10.0) * 0.628318;
    vec2 gp = (uv - 0.5) * 8.0 * u_scale;

    // FBM
    float fbmVal = fbm(gp, loopTime);

    // Voronoi
    vec2 ip = floor(gp);
    vec2 fp = fract(gp);
    float minD = 1.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 cellId = ip + g;
            float h = hash(cellId);
            float o = 0.5 + 0.4 * sin(loopTime + h * 6.28318);
            vec2 r = g + o - fp;
            float d = length(r);
            minD = min(minD, d);
        }
    }

    // Distorted grid
    vec2 gridP = gp + 0.5 * vec2(
        sin(gp.y * 2.0 + loopTime * 2.0),
        cos(gp.x * 2.0 + loopTime * 1.5)
    );
    vec2 gridVal = abs(fract(gridP) - 0.5);
    float gridLine = 1.0 - clamp(min(gridVal.x, gridVal.y) / 0.1, 0.0, 1.0);

    float heightVal = fbmVal * 0.6 + minD * 0.4;
    float result = heightVal * 0.7 + gridLine * 0.3;

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    fbm_noise: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float loopTime = mod(u_time, 10.0) * 0.628318;
    vec2 p = uv * 6.0 * u_scale;
    float result = fbm(p, loopTime);
    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    distorted_grid: `
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float loopTime = mod(u_time, 10.0) * 0.628318;
    vec2 p = (uv - 0.5) * 10.0 * u_scale;

    vec2 distort = vec2(
        sin(p.y * 0.5 + loopTime * 2.0) * 1.5 + sin(p.y * 2.0 - loopTime * 3.0) * 0.5,
        cos(p.x * 0.5 + loopTime * 1.7) * 1.5 + cos(p.x * 2.0 - loopTime * 2.5) * 0.5
    );

    vec2 gridP = p + distort;
    vec2 gridVal = abs(fract(gridP) - 0.5);
    float lines = clamp(min(gridVal.x, gridVal.y) / 0.03, 0.0, 1.0);
    float cellShade = 0.3 + 0.7 * hash(floor(gridP));
    float result = lines * cellShade;

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    // Raymarched 3D shapes
    rm_cube: `
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float d = sdBox(rp, vec3(0.8));

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    rm_sphere: `
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float bump = 0.15 * sin(rp.x * 8.0 + loopTime * 2.0) * sin(rp.y * 8.0) * sin(rp.z * 8.0);
        float d = length(rp) - 1.0 - bump;

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    rm_torus: `
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float d = sdTorus(rp, vec2(0.8, 0.3));

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    rm_octahedron: `
float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.577;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float d = sdOctahedron(rp, 1.2);

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    rm_gyroid: `
float sdGyroid(vec3 p, float scale, float thickness) {
    p *= scale;
    float g = sin(p.x) * cos(p.y) + sin(p.y) * cos(p.z) + sin(p.z) * cos(p.x);
    return abs(g) / scale - thickness;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float sphere = length(rp) - 1.5;
        float gyroid = sdGyroid(rp, 3.0, 0.1);
        float d = max(gyroid, sphere);

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`,

    rm_menger: `
float sdMenger(vec3 p) {
    float d = max(max(abs(p.x), abs(p.y)), abs(p.z)) - 1.0;
    float s = 1.0;

    for (int i = 0; i < 3; i++) {
        vec3 a = mod(p * s, 2.0) - 1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0 * abs(a));
        float c = max(r.x, max(r.y, r.z));
        d = max(d, (c - 1.0) / s);
    }
    return d;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float loopTime = mod(u_time, 10.0) * 0.628318;

    vec3 ro = vec3(0.0, 0.0, u_camera_distance);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rotMat = rotY(radians(u_rotation_y) + loopTime) * rotX(radians(u_rotation_x) + loopTime * 0.7);

    float t = 0.0;
    float result = 0.0;

    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        vec3 rp = rotMat * p;
        float d = sdMenger(rp);

        if (d < 0.001) {
            result = 1.0 - t * 0.08;
            break;
        }
        if (t > 10.0) break;
        t += max(d, 0.001);
    }

    gl_FragColor = vec4(vec3(result), 1.0);
}`
};

// Vertex shader (shared)
const VERTEX_SHADER = `
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

class GlitchCandiesPreview {
    constructor(node) {
        this.node = node;
        this.canvas = null;
        this.gl = null;
        this.program = null;
        this.animationId = null;
        this.isPlaying = true;
        this.startTime = Date.now();
        this.currentPattern = "glitch_candies";
        this.uniforms = {};
        this.capturedFrame = null;

        this.createUI();
    }

    createUI() {
        // Create container
        this.container = document.createElement("div");
        this.container.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 5px;
            background: #1a1a1a;
            border-radius: 4px;
        `;

        // Create canvas
        this.canvas = document.createElement("canvas");
        this.canvas.width = 256;
        this.canvas.height = 256;
        this.canvas.style.cssText = `
            border: 1px solid #444;
            border-radius: 4px;
            image-rendering: pixelated;
        `;
        this.container.appendChild(this.canvas);

        // Create controls
        const controls = document.createElement("div");
        controls.style.cssText = `
            display: flex;
            gap: 5px;
            margin-top: 5px;
        `;

        // Play/Pause button
        this.playBtn = document.createElement("button");
        this.playBtn.textContent = "||";
        this.playBtn.style.cssText = `
            padding: 4px 12px;
            background: #2d2d2d;
            border: 1px solid #555;
            border-radius: 3px;
            color: #fff;
            cursor: pointer;
            font-size: 12px;
        `;
        this.playBtn.onclick = () => this.togglePlay();
        controls.appendChild(this.playBtn);

        // Capture button
        this.captureBtn = document.createElement("button");
        this.captureBtn.textContent = "Capture";
        this.captureBtn.style.cssText = `
            padding: 4px 12px;
            background: #4a4a4a;
            border: 1px solid #666;
            border-radius: 3px;
            color: #fff;
            cursor: pointer;
            font-size: 12px;
        `;
        this.captureBtn.onclick = () => this.captureFrame();
        controls.appendChild(this.captureBtn);

        this.container.appendChild(controls);

        // Status text
        this.status = document.createElement("div");
        this.status.style.cssText = `
            font-size: 10px;
            color: #888;
            margin-top: 3px;
        `;
        this.status.textContent = "Live Preview";
        this.container.appendChild(this.status);

        // Initialize WebGL
        this.initGL();
    }

    initGL() {
        this.gl = this.canvas.getContext("webgl", {
            preserveDrawingBuffer: true,
            antialias: false
        });

        if (!this.gl) {
            this.status.textContent = "WebGL not supported";
            return;
        }

        // Create vertex buffer for fullscreen quad
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1, 1, 1
        ]), this.gl.STATIC_DRAW);

        this.compileShader(this.currentPattern);
        this.startAnimation();
    }

    compileShader(pattern) {
        const gl = this.gl;
        if (!gl) return;

        // Get fragment shader code
        const fragCode = GLSL_PATTERNS[pattern];
        if (!fragCode) {
            console.warn("Unknown pattern:", pattern);
            return;
        }

        // Compile vertex shader
        const vertShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertShader, VERTEX_SHADER);
        gl.compileShader(vertShader);

        if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
            console.error("Vertex shader error:", gl.getShaderInfoLog(vertShader));
            return;
        }

        // Compile fragment shader
        const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragShader, GLSL_COMMON + fragCode);
        gl.compileShader(fragShader);

        if (!gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
            console.error("Fragment shader error:", gl.getShaderInfoLog(fragShader));
            return;
        }

        // Link program
        if (this.program) {
            gl.deleteProgram(this.program);
        }

        this.program = gl.createProgram();
        gl.attachShader(this.program, vertShader);
        gl.attachShader(this.program, fragShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.error("Program link error:", gl.getProgramInfoLog(this.program));
            return;
        }

        gl.useProgram(this.program);

        // Setup vertex attribute
        const posLoc = gl.getAttribLocation(this.program, "a_position");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        // Get uniform locations
        this.uniforms = {
            time: gl.getUniformLocation(this.program, "u_time"),
            resolution: gl.getUniformLocation(this.program, "u_resolution"),
            scale: gl.getUniformLocation(this.program, "u_scale"),
            seed: gl.getUniformLocation(this.program, "u_seed"),
            cameraDistance: gl.getUniformLocation(this.program, "u_camera_distance"),
            rotationX: gl.getUniformLocation(this.program, "u_rotation_x"),
            rotationY: gl.getUniformLocation(this.program, "u_rotation_y")
        };

        this.currentPattern = pattern;
    }

    render() {
        const gl = this.gl;
        if (!gl || !this.program) return;

        const time = (Date.now() - this.startTime) / 1000;

        // Get widget values from node
        const widgets = this.node.widgets || [];
        const getWidgetValue = (name, defaultVal) => {
            const w = widgets.find(w => w.name === name);
            return w ? w.value : defaultVal;
        };

        const pattern = getWidgetValue("pattern", "glitch_candies");
        if (pattern !== this.currentPattern) {
            this.compileShader(pattern);
        }

        const scale = getWidgetValue("scale", 1.0);
        const seed = getWidgetValue("seed", 0);
        const camDist = getWidgetValue("camera_distance", 3.0);
        const rotX = getWidgetValue("rotation_x", 0.0);
        const rotY = getWidgetValue("rotation_y", 0.0);

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);

        // Set uniforms
        if (this.uniforms.time) gl.uniform1f(this.uniforms.time, time);
        if (this.uniforms.resolution) gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
        if (this.uniforms.scale) gl.uniform1f(this.uniforms.scale, scale);
        if (this.uniforms.seed) gl.uniform1i(this.uniforms.seed, seed);
        if (this.uniforms.cameraDistance) gl.uniform1f(this.uniforms.cameraDistance, camDist);
        if (this.uniforms.rotationX) gl.uniform1f(this.uniforms.rotationX, rotX);
        if (this.uniforms.rotationY) gl.uniform1f(this.uniforms.rotationY, rotY);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    startAnimation() {
        const animate = () => {
            if (this.isPlaying) {
                this.render();
            }
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }

    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playBtn.textContent = this.isPlaying ? "||" : ">";
        this.status.textContent = this.isPlaying ? "Live Preview" : "Paused";

        if (!this.isPlaying) {
            this.render(); // Render one final frame when paused
        }
    }

    captureFrame() {
        // Render at full resolution if needed
        const widgets = this.node.widgets || [];
        const getWidgetValue = (name, defaultVal) => {
            const w = widgets.find(w => w.name === name);
            return w ? w.value : defaultVal;
        };

        const targetWidth = getWidgetValue("width", 512);
        const targetHeight = getWidgetValue("height", 512);

        // Resize canvas temporarily
        const oldWidth = this.canvas.width;
        const oldHeight = this.canvas.height;

        this.canvas.width = targetWidth;
        this.canvas.height = targetHeight;

        // Render at full resolution
        this.render();

        // Get image data
        const imageData = this.canvas.toDataURL("image/png");
        this.capturedFrame = imageData;

        // Restore canvas size
        this.canvas.width = oldWidth;
        this.canvas.height = oldHeight;

        // Send to node
        this.sendCapturedFrame(imageData);

        this.status.textContent = `Captured ${targetWidth}x${targetHeight}`;
        setTimeout(() => {
            this.status.textContent = this.isPlaying ? "Live Preview" : "Paused";
        }, 2000);
    }

    async sendCapturedFrame(imageData) {
        try {
            const response = await api.fetchApi("/koshi/glitch_candies/capture", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: this.node.id,
                    image: imageData
                })
            });

            if (response.ok) {
                // Trigger node execution with captured frame
                app.queuePrompt(0, 1);
            }
        } catch (e) {
            console.warn("Failed to send captured frame:", e);
        }
    }

    destroy() {
        this.stopAnimation();
        if (this.gl && this.program) {
            this.gl.deleteProgram(this.program);
        }
    }
}

// Store preview instances
const previews = new Map();

// Register the extension
app.registerExtension({
    name: "Koshi.GlitchCandies.LivePreview",

    async nodeCreated(node) {
        if (node.comfyClass === "Koshi_GlitchCandies") {
            // Create preview widget
            const preview = new GlitchCandiesPreview(node);
            previews.set(node.id, preview);

            // Add as DOM widget
            const widget = node.addDOMWidget("preview", "customwidget", preview.container, {
                serialize: false,
                hideOnZoom: false
            });

            // Increase node size to fit preview
            node.size[0] = Math.max(node.size[0], 280);
            node.size[1] = node.size[1] + 310;

            // Update preview when widgets change
            const originalOnWidgetChange = node.onWidgetChange;
            node.onWidgetChange = function(name, value) {
                if (originalOnWidgetChange) {
                    originalOnWidgetChange.call(this, name, value);
                }
                const p = previews.get(this.id);
                if (p && !p.isPlaying) {
                    p.render(); // Re-render when paused and value changes
                }
            };
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "Koshi_GlitchCandies") {
            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                const preview = previews.get(this.id);
                if (preview) {
                    preview.destroy();
                    previews.delete(this.id);
                }
                if (originalOnRemoved) {
                    originalOnRemoved.call(this);
                }
            };
        }
    }
});

// Register API route for frame capture
api.addEventListener("koshi_glitch_capture", (data) => {
    console.log("Glitch capture received:", data);
});

// ============================================================================
// SHAPE CANDIES PREVIEW (Koshi_ShapeCandies)
// ============================================================================

const SHAPE_CANDIES_SHADER = `
precision highp float;

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_morph;
uniform int u_shape_a;
uniform int u_shape_b;
uniform float u_noise_disp;
uniform float u_noise_freq;
uniform float u_noise_overlay;
uniform float u_cam_dist;
uniform float u_rot_x;
uniform float u_rot_y;

mat3 rotX(float a) { float c=cos(a),s=sin(a); return mat3(1,0,0,0,c,-s,0,s,c); }
mat3 rotY(float a) { float c=cos(a),s=sin(a); return mat3(c,0,s,0,1,0,-s,0,c); }

float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1,311.7,74.7))) * 43758.5453); }

float fbm3d(vec3 p, float t) {
    float r = 0.0, a = 0.5;
    for(int i=0; i<4; i++) {
        r += a * (hash(p + t) * 2.0 - 1.0);
        p *= 2.0; a *= 0.5;
    }
    return r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz)-t.x, p.y);
    return length(q) - t.y;
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x+p.y+p.z-s)*0.57735027;
}

float sdGyroid(vec3 p, float s) {
    p *= s;
    float g = sin(p.x)*cos(p.y) + sin(p.y)*cos(p.z) + sin(p.z)*cos(p.x);
    return max(abs(g) - 0.1, length(p/s) - 1.5);
}

float sdCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCone(vec3 p) {
    float q = length(p.xz);
    return max(q*0.7 + p.y*0.7 - 0.5, -p.y - 1.0);
}

float sdCapsule(vec3 p, float r) {
    p.y = p.y - clamp(p.y, -0.5, 0.5);
    return length(p) - r;
}

float sdfShape(vec3 p, int shape) {
    if(shape == 0) return sdBox(p, vec3(0.8));
    if(shape == 1) return sdSphere(p, 1.0);
    if(shape == 2) return sdTorus(p, vec2(0.8, 0.3));
    if(shape == 3) return sdOctahedron(p, 1.2);
    if(shape == 4) return sdGyroid(p, 3.0);
    if(shape == 5) return sdBox(p, vec3(1.0)); // menger simplified
    if(shape == 6) return sdCylinder(p, 0.5, 1.0);
    if(shape == 7) return sdCone(p);
    if(shape == 8) return sdCapsule(p, 0.5);
    if(shape == 9) return sdOctahedron(p, 1.0); // pyramid approx
    return sdSphere(p, 1.0);
}

float map(vec3 p, float t) {
    float morph = u_morph * u_morph * (3.0 - 2.0 * u_morph);

    // Apply noise displacement
    if(u_noise_disp > 0.0) {
        float disp = fbm3d(p * u_noise_freq, t) * u_noise_disp * 0.3;
        p += normalize(p) * disp;
    }

    float d1 = sdfShape(p, u_shape_a);
    float d2 = sdfShape(p, u_shape_b);
    return mix(d1, d2, morph);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;

    vec3 ro = vec3(0.0, 0.0, u_cam_dist);
    vec3 rd = normalize(vec3(uv, -1.5));

    mat3 rot = rotY(radians(u_rot_y) + t) * rotX(radians(u_rot_x) + t * 0.7);

    float d = 0.0;
    vec3 p;
    for(int i=0; i<64; i++) {
        p = ro + rd * d;
        p = rot * p;
        float h = map(p, t);
        if(h < 0.002 || d > 10.0) break;
        d += h;
    }

    float col = d < 10.0 ? 1.0 - d * 0.08 : 0.0;

    // Noise overlay
    if(u_noise_overlay > 0.0) {
        float n = hash(vec3(uv * 5.0, t));
        col = col * (1.0 - u_noise_overlay) + col * n * u_noise_overlay + n * u_noise_overlay * 0.2;
    }

    gl_FragColor = vec4(vec3(clamp(col, 0.0, 1.0)), 1.0);
}
`;

// Shape name to index mapping
const SHAPE_INDEX = {
    "cube": 0, "sphere": 1, "torus": 2, "octahedron": 3, "gyroid": 4,
    "menger": 5, "cylinder": 6, "cone": 7, "capsule": 8, "pyramid": 9
};

class ShapeCandiesPreview {
    constructor(node) {
        this.node = node;
        this.isPlaying = true;
        this.startTime = performance.now();

        this.container = document.createElement("div");
        this.container.style.cssText = "display:flex;flex-direction:column;background:#111;padding:4px;border-radius:4px;";

        // Header
        const header = document.createElement("div");
        header.style.cssText = "display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;";
        header.innerHTML = '<span style="color:#0f0;font-size:10px;font-family:monospace;">Shape Candies</span>';

        // Controls
        const controls = document.createElement("div");
        controls.style.cssText = "display:flex;gap:4px;";

        this.playBtn = document.createElement("button");
        this.playBtn.textContent = "⏸";
        this.playBtn.style.cssText = "background:#333;border:1px solid #555;color:#0f0;padding:2px 6px;cursor:pointer;font-size:10px;";
        this.playBtn.onclick = () => this.togglePlay();
        controls.appendChild(this.playBtn);

        header.appendChild(controls);
        this.container.appendChild(header);

        // Canvas
        this.canvas = document.createElement("canvas");
        this.canvas.width = 256;
        this.canvas.height = 256;
        this.canvas.style.cssText = "border:1px solid #333;border-radius:2px;";
        this.container.appendChild(this.canvas);

        // Status
        this.status = document.createElement("div");
        this.status.style.cssText = "color:#666;font-size:9px;font-family:monospace;text-align:center;margin-top:2px;";
        this.status.textContent = "Live Preview";
        this.container.appendChild(this.status);

        this.initGL();
        this.startAnimation();
    }

    initGL() {
        this.gl = this.canvas.getContext("webgl");
        if (!this.gl) return;

        const vs = `attribute vec2 a_pos; void main(){gl_Position=vec4(a_pos,0,1);}`;
        const vShader = this.gl.createShader(this.gl.VERTEX_SHADER);
        this.gl.shaderSource(vShader, vs);
        this.gl.compileShader(vShader);

        const fShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
        this.gl.shaderSource(fShader, SHAPE_CANDIES_SHADER);
        this.gl.compileShader(fShader);

        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vShader);
        this.gl.attachShader(this.program, fShader);
        this.gl.linkProgram(this.program);

        const buf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), this.gl.STATIC_DRAW);

        const pos = this.gl.getAttribLocation(this.program, "a_pos");
        this.gl.enableVertexAttribArray(pos);
        this.gl.vertexAttribPointer(pos, 2, this.gl.FLOAT, false, 0, 0);
    }

    getWidgetValue(name, def) {
        const w = this.node.widgets?.find(w => w.name === name);
        return w ? w.value : def;
    }

    render() {
        if (!this.gl || !this.program) return;

        this.gl.useProgram(this.program);

        const t = (performance.now() - this.startTime) / 1000;
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_time"), t);
        this.gl.uniform2f(this.gl.getUniformLocation(this.program, "u_resolution"), this.canvas.width, this.canvas.height);

        const shapeA = SHAPE_INDEX[this.getWidgetValue("shape_a", "sphere")] || 1;
        const shapeB = SHAPE_INDEX[this.getWidgetValue("shape_b", "cube")] || 0;
        this.gl.uniform1i(this.gl.getUniformLocation(this.program, "u_shape_a"), shapeA);
        this.gl.uniform1i(this.gl.getUniformLocation(this.program, "u_shape_b"), shapeB);
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_morph"), this.getWidgetValue("morph_amount", 0.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_noise_disp"), this.getWidgetValue("noise_displacement", 0.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_noise_freq"), this.getWidgetValue("noise_frequency", 3.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_noise_overlay"), this.getWidgetValue("noise_overlay", 0.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_cam_dist"), this.getWidgetValue("camera_distance", 3.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_rot_x"), this.getWidgetValue("rotation_x", 0.0));
        this.gl.uniform1f(this.gl.getUniformLocation(this.program, "u_rot_y"), this.getWidgetValue("rotation_y", 0.0));

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playBtn.textContent = this.isPlaying ? "⏸" : "▶";
        this.status.textContent = this.isPlaying ? "Live Preview" : "Paused";
        if (this.isPlaying) this.startAnimation();
    }

    startAnimation() {
        const animate = () => {
            if (!this.isPlaying) return;
            this.render();
            this.animFrame = requestAnimationFrame(animate);
        };
        animate();
    }

    stopAnimation() {
        if (this.animFrame) cancelAnimationFrame(this.animFrame);
    }

    destroy() {
        this.stopAnimation();
        if (this.gl && this.program) this.gl.deleteProgram(this.program);
    }
}

// ShapeCandies functionality merged into Koshi_GlitchCandies
// 3D shape preview uses ShapeCandiesPreview class when pattern starts with "rm_" or is "shape_morph"
