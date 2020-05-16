#version 440 core

layout (location=0) in vec4 position;
layout (location=1) in ivec4 density;

out float vs_density;
flat out float vs_discard;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vs_density = density.x;
    if (density.x <= 1.0) {
        vs_discard = 1.0;
    } else {
        vs_discard = 0.0;
    }
	gl_Position = projection * view * vec4(position.xyz, 1.0);
}