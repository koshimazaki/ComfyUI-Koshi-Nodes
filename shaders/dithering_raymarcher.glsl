#version 330
// Dithering patterns based on glsl-dither by Hugh Kennedy (MIT)
// https://github.com/hughsk/glsl-dither
// Additional inspiration from paper-design/shaders
// https://github.com/paper-design/shaders

uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;  // Blue noise texture for dithering
uniform float grainAmount;     // Control dithering intensity (0.0 - 1.0)
uniform bool monoOutput;       // Output monochrome or RGB
uniform int patternType;       // Dithering pattern: 0=Blue Noise, 1=Bayer 2x2, 2=Bayer 4x4, 3=Bayer 8x8, 4=Random
uniform int shapeType;         // 3D Shape: 0=Torus, 1=Cube, 2=Sphere, 3=Dodecahedron, 4=Tetrahedron

out vec4 fragColor;

#define MAX_STEPS 100
#define MAX_DIST 100.
#define EPSILON 0.0001
#define PI 3.14159265
#define RED 4.

mat2 Rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdDodecahedron(vec3 p, float r) {
    const vec3 n = normalize(vec3(1.618034, 1.0, 0.0)); // Golden ratio
    p = abs(p);
    float d = dot(p, normalize(n.xyz)) - r;
    d = max(d, dot(p, normalize(n.zxy)) - r);
    d = max(d, dot(p, normalize(n.yzx)) - r);
    return d;
}

float sdTetrahedron(vec3 p, float r) {
    const float k = sqrt(2.0);
    p.xy = abs(p.xy);
    p.xy -= min(dot(vec2(-k, k), p.xy), 0.0) * vec2(-k, k);
    p.xz -= min(dot(vec2(-k, k), p.xz), 0.0) * vec2(-k, k);
    p.yz -= min(dot(vec2(-k, k), p.yz), 0.0) * vec2(-k, k);
    return (length(p) - r) / sqrt(3.0);
}

vec2 getDist(vec3 p, float t) {
    p.xz *= Rot(t * 5.);
    p.xy *= Rot(t * 7.);
    float scale = 1. + .2 * sin(t * 10.);
    p /= scale;

    float dist;
    if (shapeType == 1) {
        dist = sdBox(p, vec3(1.0));
    } else if (shapeType == 2) {
        dist = sdSphere(p, 1.5);
    } else if (shapeType == 3) {
        dist = sdDodecahedron(p, 1.0);
    } else if (shapeType == 4) {
        dist = sdTetrahedron(p, 1.5);
    } else {
        dist = sdTorus(p, vec2(1.2, .5));
    }

    return vec2(dist * scale, RED);
}

vec3 rayMarch(vec3 ro, vec3 rd, float t) {
    float d = 0.;
    float info = 0.;
    int steps = 0;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec2 distToClosest = getDist(ro + rd * d, t);
        steps++;
        d += abs(distToClosest.x);
        info = distToClosest.y;
        if (abs(distToClosest.x) < EPSILON || d > MAX_DIST) {
            break;
        }
    }
    return vec3(d, info, steps);
}

vec3 getNormal(vec3 p, float t) {
    vec2 e = vec2(EPSILON, 0.);
    vec3 n = getDist(p, t).x - vec3(
        getDist(p - e.xyy, t).x,
        getDist(p - e.yxy, t).x,
        getDist(p - e.yyx, t).x
    );
    return normalize(n);
}

vec3 getRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l - p),
         r = normalize(cross(vec3(0, 1, 0), f)),
         u = cross(f, r),
         c = f * z,
         i = c + uv.x * r + uv.y * u,
         d = normalize(i);
    return d;
}

// ========================================
// Dithering Functions
// Based on: https://github.com/hughsk/glsl-dither
// ========================================

// Simple hash function for random dithering
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

// Unified dithering function that selects pattern based on patternType
float applyDithering(vec2 position, float brightness, int pattern) {
    if (pattern == 1) {
        return dither2x2(position, brightness);
    } else if (pattern == 2) {
        return dither4x4(position, brightness);
    } else if (pattern == 3) {
        return dither8x8(position, brightness);
    } else if (pattern == 4) {
        // Pattern 4: Random dithering
        return ditherRandom(position, brightness);
    } else {
        // Pattern 0: Blue noise (texture-based)
        float ditherPattern = texture(iChannel0, position / 32.).x;
        return step(ditherPattern, brightness);
    }
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    float camRadius = 4.;
    vec3 ro = vec3(0, 0, -camRadius);
    float zoom = 1.100;
    float t = iTime / 5.;
    vec3 color, rm, rd = getRayDir(uv, ro, vec3(0), 1.);
    float d;

    for (int i = 0; i < 3; i++) {
        rm = rayMarch(ro, rd, t);
        d = rm[0];
        vec3 light = vec3(10, 0, 0);
        vec3 p = ro + rd * d;

        if (d < MAX_DIST) {
            vec3 n = getNormal(p, t);
            vec3 dirToLight = normalize(light - p);
            vec3 rayMarchLight = rayMarch(p + dirToLight * .06, dirToLight, t);
            float distToObstable = rayMarchLight.x;
            float distToLight = length(light - p);

            if (d < MAX_DIST) {
                // Calculate base lighting
                color[i] = .5 * (dot(n, normalize(light - p))) + .5;

                // Apply selected dithering pattern with grain control
                vec2 ditherPos = fragCoord + 8. * float(i);
                float dithered = applyDithering(ditherPos, color[i], patternType);

                // Interpolate between dithered and smooth based on grainAmount
                color[i] = mix(color[i], dithered, grainAmount);
            }
        }
        t += .01;
    }

    // Apply mono output if enabled
    if (monoOutput) {
        float mono = (color.r + color.g + color.b) / 3.0;
        fragColor = vec4(vec3(mono), 1.0);
    } else {
        fragColor = vec4(color, 1.0);
    }
}
