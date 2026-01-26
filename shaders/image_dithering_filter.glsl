#version 330
// Dithering patterns based on glsl-dither by Hugh Kennedy (MIT)
// https://github.com/hughsk/glsl-dither
// Additional inspiration from paper-design/shaders
// https://github.com/paper-design/shaders

uniform vec2 iResolution;
uniform sampler2D inputImage;      // Input image to dither
uniform vec3 colorBack;            // Background color
uniform vec3 colorFront;           // Main foreground color
uniform vec3 colorHighlight;       // Secondary foreground color (3-color mode)
uniform bool useOriginalColors;    // Use image's original colors instead of palette
uniform int patternType;           // 0=Blue Noise, 1=Bayer 2x2, 2=Bayer 4x4, 3=Bayer 8x8, 4=Random
uniform float pixelSize;           // Pixel grid size (1.0 = normal, higher = larger pixels)
uniform int colorSteps;            // Posterization levels (1-7)
uniform float ditherIntensity;     // Dithering strength (0.0-1.0)

out vec4 fragColor;

// ========================================
// Dithering Functions
// ========================================

// Hash function for random dithering
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Random dithering
float ditherRandom(vec2 position, float brightness) {
    float threshold = hash12(position);
    return brightness < threshold ? 0.0 : 1.0;
}

// Bayer 2x2 Matrix - Optimized with const array
const float bayer2x2[4] = float[4](
    0.25, 0.75,
    1.00, 0.50
);

float dither2x2(vec2 position, float brightness) {
    int x = int(mod(position.x, 2.0));
    int y = int(mod(position.y, 2.0));
    int index = x + y * 2;
    return step(bayer2x2[index], brightness);
}

// Bayer 4x4 Matrix - Optimized with const array
const float bayer4x4[16] = float[16](
    0.0625,  0.5625,  0.1875,  0.6875,
    0.8125,  0.3125,  0.9375,  0.4375,
    0.25,    0.75,    0.125,   0.625,
    1.0,     0.5,     0.875,   0.375
);

float dither4x4(vec2 position, float brightness) {
    int x = int(mod(position.x, 4.0));
    int y = int(mod(position.y, 4.0));
    int index = x + y * 4;
    return step(bayer4x4[index], brightness);
}

// Bayer 8x8 Matrix - Optimized with const array
const float bayer8x8[64] = float[64](
    0.015625, 0.515625, 0.140625, 0.640625, 0.046875, 0.546875, 0.171875, 0.671875,
    0.765625, 0.265625, 0.890625, 0.390625, 0.796875, 0.296875, 0.921875, 0.421875,
    0.203125, 0.703125, 0.078125, 0.578125, 0.234375, 0.734375, 0.109375, 0.609375,
    0.953125, 0.453125, 0.828125, 0.328125, 0.984375, 0.484375, 0.859375, 0.359375,
    0.0625,   0.5625,   0.1875,   0.6875,   0.03125,  0.53125,  0.15625,  0.65625,
    0.8125,   0.3125,   0.9375,   0.4375,   0.78125,  0.28125,  0.90625,  0.40625,
    0.25,     0.75,     0.125,    0.625,    0.21875,  0.71875,  0.09375,  0.59375,
    1.0,      0.5,      0.875,    0.375,    0.96875,  0.46875,  0.84375,  0.34375
);

float dither8x8(vec2 position, float brightness) {
    int x = int(mod(position.x, 8.0));
    int y = int(mod(position.y, 8.0));
    int index = x + y * 8;
    return step(bayer8x8[index], brightness);
}

// Apply dithering pattern
float applyDithering(vec2 position, float brightness, int pattern) {
    if (pattern == 1) {
        return dither2x2(position, brightness);
    } else if (pattern == 2) {
        return dither4x4(position, brightness);
    } else if (pattern == 3) {
        return dither8x8(position, brightness);
    } else if (pattern == 4) {
        return ditherRandom(position, brightness);
    } else {
        // Pattern 0: Blue noise - use simple hash-based for now
        return ditherRandom(position, brightness);
    }
}

// RGB to luminance
float getLuminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// Posterize color to N levels
vec3 posterize(vec3 color, int levels) {
    if (levels <= 1) return vec3(0.0);
    float steps = float(levels - 1);
    return floor(color * steps + 0.5) / steps;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = fragCoord / iResolution;

    // Apply pixelization
    vec2 pixelatedUV = uv;
    if (pixelSize > 1.0) {
        vec2 pixelCoord = floor(fragCoord / pixelSize) * pixelSize;
        pixelatedUV = pixelCoord / iResolution;
    }

    // Sample input image
    vec3 originalColor = texture(inputImage, pixelatedUV).rgb;

    // Apply posterization if using original colors
    vec3 workingColor = originalColor;
    if (useOriginalColors && colorSteps > 0) {
        workingColor = posterize(originalColor, colorSteps);
    }

    // Calculate luminance for dithering
    float luma = getLuminance(workingColor);

    // Apply dithering
    vec2 ditherPos = floor(fragCoord / pixelSize);
    float dithered = applyDithering(ditherPos, luma, patternType);

    // Mix dithered with original based on intensity
    luma = mix(luma, dithered, ditherIntensity);

    vec3 finalColor;

    if (useOriginalColors) {
        // Use original image colors with dithering applied
        finalColor = workingColor * luma;
    } else {
        // Use color palette (2-color or 3-color mode)
        if (colorSteps <= 2 || colorFront == colorHighlight) {
            // 2-color mode
            finalColor = mix(colorBack, colorFront, luma);
        } else {
            // 3-color mode
            if (luma < 0.33) {
                finalColor = colorBack;
            } else if (luma < 0.66) {
                finalColor = colorFront;
            } else {
                finalColor = colorHighlight;
            }
        }
    }

    fragColor = vec4(finalColor, 1.0);
}
