#version 440 core

layout (location=0) in vec4 vPosition;
layout (location=1) in ivec4 vDensity;

uniform mat4 projection;
uniform mat4 view;

flat out float vs_density;
flat out float vs_discard;

void main()
{
    vs_density = vDensity.x;
    if (vDensity.x <= 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
	gl_Position = projection * view * vec4(vPosition.xyz, 1.0);
}