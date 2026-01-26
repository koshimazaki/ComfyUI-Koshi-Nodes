#version 330

uniform sampler2D image;
uniform int mode; // 0: Prefilter, 1: Downsample, 2: Upsample
uniform float threshold;
uniform float knee;
uniform float intensity;

in vec2 v_uv;
out vec4 fragColor;

// Curve calculation for soft threshold
vec4 QuadraticThreshold(vec4 color, float threshold, vec3 curve) {
    float br = max(color.r, max(color.g, color.b));
    float rq = clamp(br - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;
    color *= max(rq, br - threshold) / max(br, 0.0001);
    return color;
}

void main() {
    vec2 tex_size = textureSize(image, 0);
    vec2 tex_el_size = 1.0 / tex_size;

    if (mode == 0) {
        // Prefilter (Threshold)
        vec4 color = texture(image, v_uv);
        
        // Soft threshold
        // curve = (threshold - knee, knee * 2, 0.25 / knee)
        vec3 curve = vec3(threshold - knee, knee * 2.0, 0.25 / knee);
        color = QuadraticThreshold(color, threshold, curve);
        
        fragColor = color;
    } 
    else if (mode == 1) {
        // Downsample (13-tap box filter)
        // A - B - C
        // - D - E -
        // F - G - H
        // - I - J -
        // K - L - M
        
        vec4 A = texture(image, v_uv + tex_el_size * vec2(-2, 2));
        vec4 B = texture(image, v_uv + tex_el_size * vec2(0, 2));
        vec4 C = texture(image, v_uv + tex_el_size * vec2(2, 2));
        vec4 D = texture(image, v_uv + tex_el_size * vec2(-1, 1));
        vec4 E = texture(image, v_uv + tex_el_size * vec2(1, 1));
        vec4 F = texture(image, v_uv + tex_el_size * vec2(-2, 0));
        vec4 G = texture(image, v_uv);
        vec4 H = texture(image, v_uv + tex_el_size * vec2(2, 0));
        vec4 I = texture(image, v_uv + tex_el_size * vec2(-1, -1));
        vec4 J = texture(image, v_uv + tex_el_size * vec2(1, -1));
        vec4 K = texture(image, v_uv + tex_el_size * vec2(-2, -2));
        vec4 L = texture(image, v_uv + tex_el_size * vec2(0, -2));
        vec4 M = texture(image, v_uv + tex_el_size * vec2(2, -2));

        vec4 div = (D + E + I + J) * 0.5;
        div += (A + B + G + F) * 0.125;
        div += (B + C + H + G) * 0.125;
        div += (F + G + L + K) * 0.125;
        div += (G + H + M + L) * 0.125;

        fragColor = div * 0.25; // Normalization might be slightly off in standard formula, but this is a common approx
        // Actually standard 13-tap is:
        // 0.125 * (A+B+G+F) + 0.125*(B+C+H+G) ... 
        // The above sum is:
        // (D+E+I+J)*0.5 + (A+C+K+M)*0.125 + (B+F+H+L)*0.25 + G*0.5 -> Total weight 4.0 -> divide by 4.0
        // My implementation above:
        // div sum weights: 4 * 0.5 + 4 * 0.125 * 4 = 2 + 2 = 4.
        // So divide by 4 is correct.
    }
    else if (mode == 2) {
        // Upsample (9-tap tent filter)
        float r = 1.0; // radius
        
        vec4 d = tex_el_size.xyxy * vec4(1, 1, -1, 0) * r;

        vec4 s;
        s =  texture(image, v_uv - d.xy);
        s += texture(image, v_uv - d.wy) * 2.0;
        s += texture(image, v_uv - d.zy);

        s += texture(image, v_uv + d.zw) * 2.0;
        s += texture(image, v_uv       ) * 4.0;
        s += texture(image, v_uv + d.xw) * 2.0;

        s += texture(image, v_uv + d.zy);
        s += texture(image, v_uv + d.wy) * 2.0;
        s += texture(image, v_uv + d.xy);

        fragColor = s * (1.0 / 16.0);
    }
    else if (mode == 3) {
        // Final Combine (Additive)
        // This might be done via blending in GL, but if we do it in shader:
        // We expect 'image' to be the bloom texture, and we might need another sampler for original
        // But to keep it simple, we'll just output the bloom color and use GL blending in Python
        fragColor = texture(image, v_uv) * intensity;
    }
}
