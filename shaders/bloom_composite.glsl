// Bloom Composite Shader
// Based on alien.js by Patrick Schroen
// https://github.com/alienkitty/alien.js
// MIT License

#version 330

uniform sampler2D tBlur1;
uniform sampler2D tBlur2;
uniform sampler2D tBlur3;
uniform sampler2D tBlur4;
uniform sampler2D tBlur5;
uniform float uBloomFactors[5];

in vec2 vUv;
out vec4 FragColor;

void main() {
    FragColor = uBloomFactors[0] * texture(tBlur1, vUv) +
                uBloomFactors[1] * texture(tBlur2, vUv) +
                uBloomFactors[2] * texture(tBlur3, vUv) +
                uBloomFactors[3] * texture(tBlur4, vUv) +
                uBloomFactors[4] * texture(tBlur5, vUv);
    FragColor.a = 1.0;
}
