#version 440

layout (location=0) in vec4 position;
layout (location=1) in ivec4 density;

out float vs_density;
out float vs_discard;

void main()
{
    if (density.x <= 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
    vs_density = pow(density.x, 1.0/3.0) / 900;
    gl_Position = vec4(position.xyz, 1.0);
}