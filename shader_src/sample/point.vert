#version 440 core

layout (location=0) in vec4 vPosition;

uniform mat4 projection;
uniform mat4 view;

void main()
{
	gl_Position = projection * view * vPosition;
}