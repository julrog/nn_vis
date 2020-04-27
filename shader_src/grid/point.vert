#version 440 core

layout (location=0) in vec4 vPosition;
layout (location=1) in ivec4 vDensity;

uniform mat4 projection;
uniform mat4 view;

flat out float vs_density;

void main()
{
    vs_density = vDensity.x;
	gl_Position = projection * view * vec4(vPosition.xyz, 1.0);
}