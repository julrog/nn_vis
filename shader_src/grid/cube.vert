#version 440

layout (location=0) in vec4 vPosition;
layout (location=1) in ivec4 vDensity;

out float vs_density;
out float vs_discard;


void main()
{
    if (vDensity.x == 0.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    vs_density = pow(vDensity.x, 1.0/2.0) / 500.0;
    gl_Position = vec4(vPosition.xyz, 1.0);
}