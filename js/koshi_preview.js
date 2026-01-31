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

// Glitch shader - matches Python GlitchShaderNode parameters
const GLITCH_SHADER = `
    precision mediump float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform float u_glitch_intensity;
    uniform float u_rgb_shift;
    uniform float u_time;
    uniform float u_shake_amount;
    uniform float u_noise_amount;
    uniform float u_scan_line_intensity;
    varying vec2 v_texCoord;

    float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
    }

    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float smoothstep2(float edge0, float edge1, float x) {
        float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }

    void main() {
        vec2 uv = v_texCoord;

        // Calculate glitch strength based on time interval (matching Python)
        float interval = 3.0;
        float strength = smoothstep2(interval * 0.5, interval, interval - mod(u_time, interval));
        strength *= u_glitch_intensity;

        // Shake effect
        float shakeX = (rand(vec2(u_time, 0.0)) * 2.0 - 1.0) * u_shake_amount * strength / u_resolution.x;
        float shakeY = (rand(vec2(u_time * 2.0, 0.0)) * 2.0 - 1.0) * u_shake_amount * strength / u_resolution.y;
        uv += vec2(shakeX, shakeY);

        // RGB wave based on row noise (like Python)
        float rowNoise1 = noise(vec2(0.0, uv.y * u_resolution.y * 0.01 + u_time * 400.0));
        float rowNoise2 = noise(vec2(0.0, uv.y * u_resolution.y * 0.02 + u_time * 200.0));
        float rgbWave = rowNoise1 * (2.0 + strength * 32.0) * rowNoise2 * (1.0 + strength * 4.0);

        // Periodic spikes
        float spike1 = sin(uv.y * u_resolution.y * 0.005 + u_time * 1.6);
        float spike2 = sin(uv.y * u_resolution.y * 0.005 + u_time * 2.0);
        rgbWave += step(0.9995, spike1) * 12.0;
        rgbWave += step(0.9999, spike2) * -18.0;

        // RGB difference per row
        float rgbDiff = u_rgb_shift + sin(u_time * 500.0 + uv.y * 40.0) * (20.0 * strength + 1.0);
        rgbDiff /= u_resolution.x;
        rgbWave /= u_resolution.x;

        // Sample with RGB shift
        float r = texture2D(u_image, uv + vec2(rgbWave + rgbDiff, 0.0)).r;
        float g = texture2D(u_image, uv + vec2(rgbWave, 0.0)).g;
        float b = texture2D(u_image, uv + vec2(rgbWave - rgbDiff, 0.0)).b;
        vec3 color = vec3(r, g, b);

        // Block glitch
        float blockY = floor(uv.y * 20.0);
        float blockNoise = rand(vec2(blockY, floor(u_time * 20.0) * 200.0));
        float blockMask = step(1.0 - (0.12 + strength * 0.3), blockNoise);
        if (blockMask > 0.5) {
            float blockShift = sin(floor(u_time * 20.0) * 200.0) * 20.0 / u_resolution.x;
            color.r = texture2D(u_image, uv + vec2(blockShift + rgbDiff, 0.0)).r;
            color.g = texture2D(u_image, uv + vec2(blockShift, 0.0)).g;
            color.b = texture2D(u_image, uv + vec2(blockShift - rgbDiff, 0.0)).b;
        }

        // White noise overlay
        float whiteNoise = (rand(uv * u_resolution + u_time) * 2.0 - 1.0) * (u_noise_amount + strength * u_noise_amount);
        color += whiteNoise;

        // Scan lines
        float scanLine = sin(uv.y * u_resolution.y * 1200.0 / u_resolution.y * 6.28318);
        scanLine = (scanLine + 1.0) * 0.5 * (u_scan_line_intensity + strength * 0.2);
        color -= scanLine;

        gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
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

// OLED Screen Emulation with full dithering support
// Uses exact SIDKIT Bayer matrices for consistency with React Three Fiber version
// Matches KoshiOLEDScreen Python node parameters
const OLED_EMULATION_SHADER = `
    precision highp float;
    uniform sampler2D u_image;
    uniform vec2 u_resolution;
    uniform vec2 u_screenSize;      // e.g., 256x128
    uniform float u_bitDepth;       // 1, 2, 4, or 8
    uniform float u_pixelGap;       // 0.0 - 0.5
    uniform float u_brightness;     // -1.0 to 1.0
    uniform float u_contrast;       // 0.5 to 2.0
    uniform float u_ditherType;     // 0=none, 1=bayer2x2, 2=bayer4x4, 3=bayer8x8, 4=floyd_steinberg
    uniform float u_ditherStrength; // 0.0 - 1.0
    uniform float u_colorMode;      // 0=grayscale, 1=green, 2=blue, 3=amber, 4=white, 5=yellow, 6=rgb
    uniform float u_bloomEnabled;   // 0 or 1
    uniform float u_bloomIntensity; // 0.0 - 1.0
    uniform float u_invert;         // 0 or 1
    varying vec2 v_texCoord;

    // Bayer 2x2 matrix (SIDKIT compatible)
    float bayer2(vec2 pos) {
        int x = int(mod(pos.x, 2.0));
        int y = int(mod(pos.y, 2.0));
        int idx = y * 2 + x;
        // [0, 2, 3, 1] / 4.0
        if (idx == 0) return 0.0;
        if (idx == 1) return 2.0 / 4.0;
        if (idx == 2) return 3.0 / 4.0;
        return 1.0 / 4.0;
    }

    // Bayer 4x4 matrix - exact SIDKIT values
    float bayer4(vec2 pos) {
        int x = int(mod(pos.x, 4.0));
        int y = int(mod(pos.y, 4.0));
        int idx = y * 4 + x;

        // Exact SIDKIT Bayer 4x4 matrix
        if (idx == 0) return 0.0 / 16.0;
        if (idx == 1) return 8.0 / 16.0;
        if (idx == 2) return 2.0 / 16.0;
        if (idx == 3) return 10.0 / 16.0;
        if (idx == 4) return 12.0 / 16.0;
        if (idx == 5) return 4.0 / 16.0;
        if (idx == 6) return 14.0 / 16.0;
        if (idx == 7) return 6.0 / 16.0;
        if (idx == 8) return 3.0 / 16.0;
        if (idx == 9) return 11.0 / 16.0;
        if (idx == 10) return 1.0 / 16.0;
        if (idx == 11) return 9.0 / 16.0;
        if (idx == 12) return 15.0 / 16.0;
        if (idx == 13) return 7.0 / 16.0;
        if (idx == 14) return 13.0 / 16.0;
        return 5.0 / 16.0;
    }

    // Bayer 8x8 matrix - exact SIDKIT values
    float bayer8(vec2 pos) {
        int x = int(mod(pos.x, 8.0));
        int y = int(mod(pos.y, 8.0));
        int idx = y * 8 + x;

        // SIDKIT Bayer 8x8 - row by row
        // Row 0
        if (idx == 0) return 0.0 / 64.0;
        if (idx == 1) return 32.0 / 64.0;
        if (idx == 2) return 8.0 / 64.0;
        if (idx == 3) return 40.0 / 64.0;
        if (idx == 4) return 2.0 / 64.0;
        if (idx == 5) return 34.0 / 64.0;
        if (idx == 6) return 10.0 / 64.0;
        if (idx == 7) return 42.0 / 64.0;
        // Row 1
        if (idx == 8) return 48.0 / 64.0;
        if (idx == 9) return 16.0 / 64.0;
        if (idx == 10) return 56.0 / 64.0;
        if (idx == 11) return 24.0 / 64.0;
        if (idx == 12) return 50.0 / 64.0;
        if (idx == 13) return 18.0 / 64.0;
        if (idx == 14) return 58.0 / 64.0;
        if (idx == 15) return 26.0 / 64.0;
        // Row 2
        if (idx == 16) return 12.0 / 64.0;
        if (idx == 17) return 44.0 / 64.0;
        if (idx == 18) return 4.0 / 64.0;
        if (idx == 19) return 36.0 / 64.0;
        if (idx == 20) return 14.0 / 64.0;
        if (idx == 21) return 46.0 / 64.0;
        if (idx == 22) return 6.0 / 64.0;
        if (idx == 23) return 38.0 / 64.0;
        // Row 3
        if (idx == 24) return 60.0 / 64.0;
        if (idx == 25) return 28.0 / 64.0;
        if (idx == 26) return 52.0 / 64.0;
        if (idx == 27) return 20.0 / 64.0;
        if (idx == 28) return 62.0 / 64.0;
        if (idx == 29) return 30.0 / 64.0;
        if (idx == 30) return 54.0 / 64.0;
        if (idx == 31) return 22.0 / 64.0;
        // Row 4
        if (idx == 32) return 3.0 / 64.0;
        if (idx == 33) return 35.0 / 64.0;
        if (idx == 34) return 11.0 / 64.0;
        if (idx == 35) return 43.0 / 64.0;
        if (idx == 36) return 1.0 / 64.0;
        if (idx == 37) return 33.0 / 64.0;
        if (idx == 38) return 9.0 / 64.0;
        if (idx == 39) return 41.0 / 64.0;
        // Row 5
        if (idx == 40) return 51.0 / 64.0;
        if (idx == 41) return 19.0 / 64.0;
        if (idx == 42) return 59.0 / 64.0;
        if (idx == 43) return 27.0 / 64.0;
        if (idx == 44) return 49.0 / 64.0;
        if (idx == 45) return 17.0 / 64.0;
        if (idx == 46) return 57.0 / 64.0;
        if (idx == 47) return 25.0 / 64.0;
        // Row 6
        if (idx == 48) return 15.0 / 64.0;
        if (idx == 49) return 47.0 / 64.0;
        if (idx == 50) return 7.0 / 64.0;
        if (idx == 51) return 39.0 / 64.0;
        if (idx == 52) return 13.0 / 64.0;
        if (idx == 53) return 45.0 / 64.0;
        if (idx == 54) return 5.0 / 64.0;
        if (idx == 55) return 37.0 / 64.0;
        // Row 7
        if (idx == 56) return 63.0 / 64.0;
        if (idx == 57) return 31.0 / 64.0;
        if (idx == 58) return 55.0 / 64.0;
        if (idx == 59) return 23.0 / 64.0;
        if (idx == 60) return 61.0 / 64.0;
        if (idx == 61) return 29.0 / 64.0;
        if (idx == 62) return 53.0 / 64.0;
        return 21.0 / 64.0;
    }

    // Pseudo-random noise for Floyd-Steinberg approximation (SIDKIT style)
    float noise(vec2 pos) {
        return fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
    }

    // Quantize value to N levels
    float quantize(float value, float levels) {
        float step = 1.0 / (levels - 1.0);
        return floor(value / step + 0.5) * step;
    }

    // Apply dithering - SIDKIT compatible algorithm
    float applyDither(float gray, vec2 pixelPos, float levels) {
        float ditherThreshold = 0.0;

        if (u_ditherType < 0.5) {
            // No dithering - simple quantization
            return quantize(gray, levels);
        } else if (u_ditherType < 1.5) {
            // Bayer 2x2
            ditherThreshold = bayer2(pixelPos);
        } else if (u_ditherType < 2.5) {
            // Bayer 4x4 (SIDKIT default)
            ditherThreshold = bayer4(pixelPos);
        } else if (u_ditherType < 3.5) {
            // Bayer 8x8
            ditherThreshold = bayer8(pixelPos);
        } else {
            // Floyd-Steinberg approximation with noise
            ditherThreshold = noise(pixelPos);
        }

        // Apply dither with strength control (SIDKIT algorithm)
        float ditherOffset = (ditherThreshold - 0.5) * u_ditherStrength;

        if (levels <= 2.0) {
            // 1-bit: threshold comparison
            float threshold = 0.5 + ditherOffset * 0.5;
            return gray > threshold ? 1.0 : 0.0;
        } else if (levels <= 4.0) {
            // 2-bit: 4 levels
            float dithered = gray + ditherOffset * 0.25;
            return quantize(clamp(dithered, 0.0, 1.0), 4.0);
        } else if (levels <= 16.0) {
            // 4-bit: 16 levels
            float dithered = gray + ditherOffset * 0.0625;
            return quantize(clamp(dithered, 0.0, 1.0), 16.0);
        } else {
            // 8-bit: 256 levels (minimal dithering effect)
            float dithered = gray + ditherOffset * 0.004;
            return quantize(clamp(dithered, 0.0, 1.0), 256.0);
        }
    }

    // Apply color mode tint
    vec3 applyColorMode(float gray) {
        vec3 color;

        if (u_colorMode < 0.5) {
            // Grayscale - slight blue tint for OLED authenticity
            color = vec3(gray * 0.95, gray, gray * 1.02);
        } else if (u_colorMode < 1.5) {
            // Green monochrome (classic terminal/Nokia)
            color = vec3(gray * 0.1, gray, gray * 0.1);
        } else if (u_colorMode < 2.5) {
            // Blue monochrome
            color = vec3(gray * 0.2, gray * 0.4, gray);
        } else if (u_colorMode < 3.5) {
            // Amber monochrome (vintage display)
            color = vec3(gray, gray * 0.6, gray * 0.1);
        } else if (u_colorMode < 4.5) {
            // White (pure, no tint)
            color = vec3(gray);
        } else if (u_colorMode < 5.5) {
            // Yellow (warm)
            color = vec3(gray, gray * 0.9, gray * 0.3);
        } else {
            // RGB passthrough
            color = vec3(gray);
        }

        return color;
    }

    // Bloom/glow effect for OLED emulation
    vec3 applyBloom(vec2 uv, vec3 baseColor, float gray) {
        if (u_bloomEnabled < 0.5 || gray < 0.3) return baseColor;

        vec2 texelSize = 1.0 / u_screenSize;
        vec3 bloom = vec3(0.0);
        float samples = 0.0;

        // Sample surrounding pixels for glow (5x5 kernel)
        for (float x = -2.0; x <= 2.0; x += 1.0) {
            for (float y = -2.0; y <= 2.0; y += 1.0) {
                vec2 offset = vec2(x, y) * texelSize * 2.0;
                vec4 s = texture2D(u_image, uv + offset);
                float sg = dot(s.rgb, vec3(0.299, 0.587, 0.114));
                if (sg > 0.4) {
                    bloom += applyColorMode(sg);
                    samples += 1.0;
                }
            }
        }

        if (samples > 0.0) {
            bloom /= samples;
            return baseColor + bloom * u_bloomIntensity * 0.4;
        }
        return baseColor;
    }

    void main() {
        vec2 screenUV = v_texCoord;
        vec2 pixelCoord = floor(screenUV * u_screenSize);
        vec2 pixelUV = (pixelCoord + 0.5) / u_screenSize;

        // Sample and convert to grayscale
        vec4 color = texture2D(u_image, pixelUV);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Apply brightness/contrast
        gray = (gray - 0.5) * u_contrast + 0.5 + u_brightness;
        gray = clamp(gray, 0.0, 1.0);

        // Apply invert if enabled
        if (u_invert > 0.5) {
            gray = 1.0 - gray;
        }

        // Calculate levels from bit depth
        float levels = pow(2.0, u_bitDepth);

        // Apply dithering
        gray = applyDither(gray, pixelCoord, levels);

        // Apply color mode
        vec3 oledColor = applyColorMode(gray);

        // Apply bloom if enabled
        oledColor = applyBloom(pixelUV, oledColor, gray);

        // OLED pixel grid effect
        vec2 pixelPos = fract(screenUV * u_screenSize);
        float pixelMask = 1.0;

        if (u_pixelGap > 0.001) {
            float gap = u_pixelGap;
            // Create pixel separation - darker at edges (OLED subpixel effect)
            float edgeX = smoothstep(0.0, gap, pixelPos.x) * smoothstep(0.0, gap, 1.0 - pixelPos.x);
            float edgeY = smoothstep(0.0, gap, pixelPos.y) * smoothstep(0.0, gap, 1.0 - pixelPos.y);
            pixelMask = edgeX * edgeY;

            // OLED true black in gaps
            pixelMask = mix(0.02, 1.0, pixelMask);
        }

        // OLED: true black when pixel is off
        if (gray < 0.01) {
            oledColor = vec3(0.0);
            pixelMask = 1.0;
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
            
            // Dither node (universal)
            "Koshi_Dither": {
                effect: "dither_bayer",
                title: "Dither Preview",
                paramMap: {
                    technique: "technique",
                    levels: "levels",
                    bayer_size: "bayer_size",
                    dot_size: "dot_size",
                    dot_angle: "angle"
                },
                // Dynamic effect selection based on technique widget
                dynamicEffect: {
                    widget: "technique",
                    mapping: {
                        "bayer": "dither_bayer",
                        "floyd_steinberg": "dither_fs",
                        "atkinson": "dither_fs",
                        "halftone": "dither_halftone",
                        "none": "passthrough"
                    }
                }
            },

            // Effects nodes
            "Koshi_Glitch": {
                effect: "glitch",
                title: "Glitch Preview",
                animated: true,
                paramMap: {
                    glitch_intensity: "glitch_intensity",
                    rgb_shift: "rgb_shift",
                    time: "time",
                    shake_amount: "shake_amount",
                    noise_amount: "noise_amount",
                    scan_line_intensity: "scan_line_intensity"
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

            // Unified Effects node
            "Koshi_Effects": {
                effect: "glitch",
                title: "Effects Preview",
                animated: true,
                dynamicEffect: {
                    widget: "effect_type",
                    mapping: {
                        "dither": "dither_bayer",
                        "bloom": "bloom",
                        "glitch": "glitch",
                        "hologram": "glitch",
                        "video_glitch": "glitch",
                        "scanlines": "dither_bayer",
                        "chromatic": "glitch"
                    }
                },
                paramMap: {
                    intensity: "glitch_intensity",
                    rgb_shift: "rgb_shift",
                    bloom_threshold: "threshold",
                    dither_levels: "levels"
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
// SIDKIT OLED SCREEN PREVIEW NODE (Koshi_OLEDScreen)
// ============================================================================

// Color mode mapping for the shader (WebGL preview only)
const OLED_COLOR_MODES = {
    "grayscale": 0,
    "green_mono": 1,
    "blue_mono": 2,
    "amber_mono": 3,
    "white_mono": 4,
    "yellow_mono": 5,
};

// Screen size presets
const OLED_SCREEN_SIZES = {
    "SSD1363 256x128": [256, 128],
    "SSD1306 128x64": [128, 64],
    "SSD1306 128x32": [128, 32],
    "Custom": null, // Use custom_width/custom_height
};

app.registerExtension({
    name: "Koshi.OLEDScreenPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Target SIDKIT OLED screen nodes (old and new names)
        if (nodeData.name !== "Koshi_OLEDScreen" && nodeData.name !== "SIDKIT_OLEDScreen") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const widget = new KoshiPreviewWidget(this, "oled", {
                title: "SIDKIT OLED Display",
                width: 256,
                height: 128,
                oledBezel: true,
                statusText: "256×128 | grayscale",
            });

            // Default OLED params - viewer mode (WebGL shows display simulation)
            widget.setParams({
                screenSize: [256, 128],
                bitDepth: 8,           // Full detail pass-through
                pixelGap: 0.15,
                brightness: 0.0,
                contrast: 1.0,
                ditherType: 0,         // No dithering in viewer
                ditherStrength: 0.0,
                colorMode: 0,          // Grayscale tint
                bloomEnabled: 0,
                bloomIntensity: 0.3,
                invert: 0,
            });

            const domWidget = this.addDOMWidget("oled_preview", "preview", widget.getElement(), {
                serialize: false,
                hideOnZoom: false,
            });

            domWidget.computeSize = (width) => [width, widget.getHeight() + 40];
            this.oledPreview = widget;

            // Video playback state
            this.videoFrames = [];
            this.currentFrame = 0;
            this.isPlaying = false;
            this.playbackInterval = null;
            this.fps = 12;

            // Add video controls
            this._createVideoControls();

            // Initial param sync
            this._syncOLEDPreviewParams();
        };

        // Create video playback controls
        nodeType.prototype._createVideoControls = function() {
            if (!this.oledPreview) return;

            const container = this.oledPreview.getElement();

            // Create controls bar
            const controls = document.createElement("div");
            controls.style.cssText = `
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                padding: 4px 8px;
                background: #1a1a1a;
                border-top: 1px solid #333;
                font-family: monospace;
                font-size: 10px;
                color: #888;
            `;

            // Play/Pause button
            const playBtn = document.createElement("button");
            playBtn.textContent = "▶";
            playBtn.title = "Play/Pause";
            playBtn.style.cssText = `
                background: #333;
                border: 1px solid #555;
                color: #0f0;
                padding: 2px 8px;
                cursor: pointer;
                font-size: 10px;
                border-radius: 3px;
            `;
            playBtn.onclick = () => this._togglePlayback();

            // Frame counter
            const frameInfo = document.createElement("span");
            frameInfo.textContent = "0/0";
            frameInfo.style.minWidth = "50px";
            frameInfo.style.textAlign = "center";

            // Frame slider
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = 0;
            slider.max = 0;
            slider.value = 0;
            slider.style.cssText = "width: 100px; accent-color: #0f0;";
            slider.oninput = () => {
                this.currentFrame = parseInt(slider.value);
                this._showFrame(this.currentFrame);
            };

            // FPS control
            const fpsLabel = document.createElement("span");
            fpsLabel.textContent = "FPS:";
            const fpsInput = document.createElement("input");
            fpsInput.type = "number";
            fpsInput.min = 1;
            fpsInput.max = 60;
            fpsInput.value = this.fps;
            fpsInput.style.cssText = "width: 35px; background: #222; border: 1px solid #444; color: #0f0; text-align: center;";
            fpsInput.onchange = () => {
                this.fps = Math.max(1, Math.min(60, parseInt(fpsInput.value) || 12));
                if (this.isPlaying) {
                    this._stopPlayback();
                    this._startPlayback();
                }
            };

            controls.appendChild(playBtn);
            controls.appendChild(frameInfo);
            controls.appendChild(slider);
            controls.appendChild(fpsLabel);
            controls.appendChild(fpsInput);

            container.appendChild(controls);

            this._videoControls = { playBtn, frameInfo, slider, fpsInput };
        };

        nodeType.prototype._togglePlayback = function() {
            if (this.isPlaying) {
                this._stopPlayback();
            } else {
                this._startPlayback();
            }
        };

        nodeType.prototype._startPlayback = function() {
            if (this.videoFrames.length < 2) return;

            this.isPlaying = true;
            this._videoControls.playBtn.textContent = "⏸";

            this.playbackInterval = setInterval(() => {
                this.currentFrame = (this.currentFrame + 1) % this.videoFrames.length;
                this._showFrame(this.currentFrame);
            }, 1000 / this.fps);
        };

        nodeType.prototype._stopPlayback = function() {
            this.isPlaying = false;
            this._videoControls.playBtn.textContent = "▶";

            if (this.playbackInterval) {
                clearInterval(this.playbackInterval);
                this.playbackInterval = null;
            }
        };

        nodeType.prototype._showFrame = function(index) {
            if (!this.videoFrames[index]) return;

            this.oledPreview.setImage(this.videoFrames[index]);
            this._videoControls.slider.value = index;
            this._videoControls.frameInfo.textContent = `${index + 1}/${this.videoFrames.length}`;
        };

        nodeType.prototype._loadVideoFrames = function(images) {
            this._stopPlayback();

            this.videoFrames = images.map(img =>
                `/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder || "")}`
            );

            this.currentFrame = 0;
            this._videoControls.slider.max = Math.max(0, this.videoFrames.length - 1);
            this._videoControls.slider.value = 0;

            if (this.videoFrames.length > 0) {
                this._showFrame(0);
            }

            // Auto-play if video (more than 1 frame)
            if (this.videoFrames.length > 1) {
                this._startPlayback();
            }
        };

        // Sync all widget values to preview params
        nodeType.prototype._syncOLEDPreviewParams = function() {
            if (!this.oledPreview) return;

            // Default params - viewer mode (no dithering in preview, just display simulation)
            const params = {
                screenSize: [256, 128],
                bitDepth: 8,           // Show full detail in preview
                pixelGap: 0.15,
                brightness: 0.0,
                contrast: 1.0,
                ditherType: 0,         // No dithering - pass through
                ditherStrength: 0.0,
                colorMode: 0,
                bloomEnabled: 0,
                bloomIntensity: 0.3,
                invert: 0,
            };

            let customWidth = 256;
            let customHeight = 128;

            for (const w of this.widgets || []) {
                switch (w.name) {
                    case "screen_preset":
                        const size = OLED_SCREEN_SIZES[w.value];
                        if (size) params.screenSize = size;
                        break;
                    case "custom_width":
                        customWidth = w.value;
                        break;
                    case "custom_height":
                        customHeight = w.value;
                        break;
                    case "color_mode":
                        params.colorMode = OLED_COLOR_MODES[w.value] ?? 0;
                        break;
                    case "pixel_gap":
                        params.pixelGap = w.value;
                        break;
                    case "show_pixel_grid":
                        if (!w.value) params.pixelGap = 0;
                        break;
                    case "bloom_glow":
                        params.bloomEnabled = w.value ? 1 : 0;
                        break;
                    case "bloom_intensity":
                        params.bloomIntensity = w.value ?? 0.3;
                        break;
                }
            }

            // Handle Custom preset
            const preset = this.widgets?.find(w => w.name === "screen_preset")?.value;
            if (preset === "Custom") {
                params.screenSize = [customWidth, customHeight];
            }

            // Update status text
            const colorMode = this.widgets?.find(w => w.name === "color_mode")?.value || "grayscale";
            const sizeStr = `${params.screenSize[0]}×${params.screenSize[1]}`;
            this.oledPreview.status.textContent = `${sizeStr} | ${colorMode}`;

            this.oledPreview.setParams(params);
        };

        // Update preview when any widget changes
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(name, value, old_value, widget) {
            if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            this._syncOLEDPreviewParams?.();
        };

        // Also handle property changes
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function(name, value) {
            if (onPropertyChanged) onPropertyChanged.apply(this, arguments);
            this._syncOLEDPreviewParams?.();
        };

        // Handle execution result - load all frames for video playback
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (this.oledPreview && message?.images?.length > 0) {
                // Load all frames for video playback
                this._loadVideoFrames(message.images);
            }
        };

        // Try to get input image from connected node
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (onConnectionsChange) onConnectionsChange.apply(this, arguments);

            // When input connection changes, try to get preview from connected node
            if (type === 1 && index === 0) { // Input slot 0 = images
                this._tryLoadInputPreview?.();
            }
        };

        nodeType.prototype._tryLoadInputPreview = function() {
            if (!this.oledPreview || !this.inputs || !this.inputs[0]) return;

            const link = this.inputs[0].link;
            if (!link) return;

            const linkInfo = app.graph.links[link];
            if (!linkInfo) return;

            const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
            if (!sourceNode || !sourceNode.imgs || sourceNode.imgs.length === 0) return;

            // Get the image URL from the source node
            const img = sourceNode.imgs[0];
            if (img && img.src) {
                this.oledPreview.setImage(img.src);
            }
        };
    }
});

console.log("[Koshi] Live Preview + OLED Emulation loaded");
